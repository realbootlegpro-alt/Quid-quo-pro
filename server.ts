import express from "express";
import fs from "fs";
import path from "path";
import multer from "multer";
import { spawn } from "child_process";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const ROOT_DIR = path.resolve(__dirname);
const IMPORT_DIR = path.join(ROOT_DIR, "import_container");
const INDEX_DIR = path.join(IMPORT_DIR, "_index");
const PROFILES_DIR = path.join(INDEX_DIR, "profiles");
const BATCHES_DIR = path.join(ROOT_DIR, "batches");
const PROFILER_SCRIPT = path.join(ROOT_DIR, "scripts", "profile_container.py");
const BOOTSTRAP_SCRIPT = path.join(ROOT_DIR, "scripts", "bootstrap_runtime.sh");
const PYDEPS_DIR = path.join(ROOT_DIR, ".pydeps");
const LOCAL_BLENDER_BIN = path.join(ROOT_DIR, ".tools", "blender", "blender");

const TMP_UPLOADS_DIR = path.join(IMPORT_DIR, "_tmp");

for (const dir of [
  path.join(IMPORT_DIR, "characters"),
  path.join(IMPORT_DIR, "props"),
  PROFILES_DIR,
  TMP_UPLOADS_DIR,
  BATCHES_DIR,
]) {
  fs.mkdirSync(dir, { recursive: true });
}

// Clear stale partial uploads left from interrupted runs
try {
  for (const entry of fs.readdirSync(TMP_UPLOADS_DIR)) {
    fs.rmSync(path.join(TMP_UPLOADS_DIR, entry), { recursive: true, force: true });
  }
} catch {
  // Ignore startup temp cleanup errors
}

const SAFE_SEGMENT_RE = /^[a-zA-Z0-9][a-zA-Z0-9._-]{0,120}$/;
const UPLOAD_ID_RE = /^[a-zA-Z0-9_-]{8,64}$/;
const MAX_UPLOAD_BYTES = 100 * 1024 * 1024;
const MAX_CHUNK_BYTES = 8 * 1024 * 1024;

function assertSafeSegment(val: unknown, label: string): string {
  if (typeof val !== "string" || !SAFE_SEGMENT_RE.test(val) || val.includes("..")) {
    throw new Error(`Invalid ${label}`);
  }
  return val;
}

function assertUploadId(val: unknown): string {
  if (typeof val !== "string" || !UPLOAD_ID_RE.test(val)) {
    throw new Error("Invalid uploadId");
  }
  return val;
}

function resolveInside(baseDir: string, ...parts: string[]): string {
  const baseResolved = path.resolve(baseDir);
  const target = path.resolve(baseResolved, ...parts);
  if (target !== baseResolved && !target.startsWith(baseResolved + path.sep)) {
    throw new Error("Path escapes workspace boundary");
  }
  return target;
}

function sanitizeFilename(originalName: string): string {
  const rawBase = path.basename(originalName || "").trim();
  const rawExt = path.extname(rawBase).toLowerCase();
  const rawStem = path.basename(rawBase, path.extname(rawBase));
  let cleanStem = rawStem
    .replace(/[^a-zA-Z0-9._-]/g, "_")
    .replace(/^[^a-zA-Z0-9]+/, "");
  if (!cleanStem) {
    cleanStem = "asset";
  }
  const cleanExt = rawExt.replace(/[^a-z0-9.]/g, "");
  const maxStemLen = Math.max(1, 120 - cleanExt.length);
  if (cleanStem.length > maxStemLen) {
    cleanStem = cleanStem.slice(0, maxStemLen);
  }
  const finalName = `${cleanStem}${cleanExt}`;
  if (!SAFE_SEGMENT_RE.test(finalName) || finalName.includes("..")) {
    throw new Error("Invalid filename");
  }
  return finalName;
}

async function assembleChunksAtomically(
  chunkDir: string,
  totalChunks: number,
  destDir: string,
  finalFilename: string
): Promise<number> {
  const tempDest = resolveInside(
    destDir,
    `.assembling_${Date.now()}_${Math.random().toString(36).slice(2, 10)}`
  );
  let totalBytes = 0;
  try {
    const outFd = await fs.promises.open(tempDest, "w");
    try {
      for (let i = 0; i < totalChunks; i++) {
        const partPath = resolveInside(
          chunkDir,
          `part_${String(i).padStart(4, "0")}`
        );
        if (!fs.existsSync(partPath)) {
          throw new Error(`Missing upload chunk ${i + 1} of ${totalChunks}`);
        }
        const buf = await fs.promises.readFile(partPath);
        totalBytes += buf.length;
        if (totalBytes > MAX_UPLOAD_BYTES) {
          throw new Error("File exceeds 100 MB maximum limit");
        }
        await outFd.write(buf);
      }
    } finally {
      await outFd.close();
    }

    const finalPath = resolveInside(destDir, finalFilename);
    await fs.promises.rename(tempDest, finalPath);
    return totalBytes;
  } catch (err) {
    await fs.promises.rm(tempDest, { force: true }).catch(() => {});
    throw err;
  }
}

interface CatalogEntry {
  id: string;
  category: string;
  source_file: string;
  files?: {
    source?: string;
    norm_blend?: string;
    preview_image?: string;
    meta_json?: string;
  };
  dimensions_m?: number[];
  is_container?: boolean;
}

function readCatalog(): {
  characters: Record<string, CatalogEntry>;
  props: Record<string, CatalogEntry>;
} {
  const catalogPath = path.join(INDEX_DIR, "asset_catalog.json");
  if (!fs.existsSync(catalogPath)) {
    return { characters: {}, props: {} };
  }
  try {
    const raw = JSON.parse(fs.readFileSync(catalogPath, "utf-8"));
    return {
      characters: raw.characters || {},
      props: raw.props || {},
    };
  } catch {
    return { characters: {}, props: {} };
  }
}

let profilerRunning = false;

async function startServer() {
  const app = express();
  app.use(express.json());
  app.use((_req, res, next) => {
    res.setHeader("X-Content-Type-Options", "nosniff");
    next();
  });

  const assetStorage = multer.diskStorage({
    destination: (req, _file, cb) => {
      try {
        const category = req.params.category;
        if (category !== "characters" && category !== "props") {
          return cb(new Error("Category must be characters or props"), "");
        }
        const dest = resolveInside(IMPORT_DIR, category);
        fs.mkdirSync(dest, { recursive: true });
        cb(null, dest);
      } catch (err) {
        cb(err as Error, "");
      }
    },
    filename: (_req, file, cb) => {
      try {
        cb(null, sanitizeFilename(file.originalname));
      } catch (err) {
        cb(err as Error, "");
      }
    },
  });

  const uploadAssets = multer({
    storage: assetStorage,
    limits: { fileSize: 100 * 1024 * 1024 },
    fileFilter: (_req, file, cb) => {
      const ext = path.extname(file.originalname).toLowerCase();
      if (ext === ".glb" || ext === ".blend") {
        cb(null, true);
      } else {
        cb(new Error("Only .glb and .blend files are allowed"));
      }
    },
  });

  app.get("/api/import-container/:category", (req, res) => {
    try {
      const category = req.params.category;
      if (category !== "characters" && category !== "props") {
        return res.status(400).json({ error: "Invalid category" });
      }
      const catDir = resolveInside(IMPORT_DIR, category);
      const catalog = readCatalog();
      const catCatalog = catalog[category] || {};
      const bySourceFile = new Map<string, CatalogEntry>();
      for (const entry of Object.values(catCatalog)) {
        if (entry.source_file) {
          bySourceFile.set(entry.source_file, entry);
        }
      }

      const entries = fs.readdirSync(catDir, { withFileTypes: true });
      const files = entries
        .filter((e) => e.isFile() && /\.(glb|blend)$/i.test(e.name))
        .sort((a, b) => a.name.localeCompare(b.name))
        .map((e) => {
          const full = resolveInside(catDir, e.name);
          const stat = fs.statSync(full);
          const catEntry = bySourceFile.get(e.name);
          const previewRel = catEntry?.files?.preview_image || null;
          const previewExists = previewRel
            ? fs.existsSync(resolveInside(IMPORT_DIR, previewRel))
            : false;

          return {
            name: e.name,
            size: stat.size,
            mtimeMs: stat.mtimeMs,
            status: catEntry ? "Profiled" : "Unprofiled",
            assetId: catEntry?.id || null,
            dimensions_m: catEntry?.dimensions_m || null,
            previewFile: previewExists && previewRel ? path.basename(previewRel) : null,
          };
        });

      res.json({ category, files, profilerRunning });
    } catch (err) {
      res.status(500).json({ error: (err as Error).message });
    }
  });

  const uploadChunkMulter = multer({
    storage: multer.memoryStorage(),
    limits: { fileSize: MAX_CHUNK_BYTES, files: 1 },
  });

  app.post("/api/import-container/:category/upload", (req, res) => {
    uploadAssets.array("files", 20)(req, res, (err) => {
      if (err) {
        return res.status(400).json({ error: err.message });
      }
      const uploaded = ((req.files as Express.Multer.File[]) || []).map((f) => ({
        name: f.filename,
        size: f.size,
      }));
      res.json({ uploaded });
    });
  });

  app.post("/api/import-container/:category/upload/chunk", (req, res) => {
    uploadChunkMulter.single("chunk")(req, res, async (err) => {
      if (err) {
        return res.status(400).json({ error: err.message });
      }
      let chunkDir: string | null = null;
      try {
        const category = req.params.category;
        if (category !== "characters" && category !== "props") {
          return res.status(400).json({ error: "Category must be characters or props" });
        }
        const uploadId = assertUploadId(req.body?.uploadId);
        const safeName = sanitizeFilename(req.body?.filename);
        const ext = path.extname(safeName).toLowerCase();
        if (ext !== ".glb" && ext !== ".blend") {
          return res.status(400).json({ error: "Only .glb and .blend files are allowed" });
        }

        const chunkIndex = Number(req.body?.chunkIndex);
        const totalChunks = Number(req.body?.totalChunks);
        const totalSize = Number(req.body?.totalSize);

        if (
          !Number.isInteger(chunkIndex) ||
          !Number.isInteger(totalChunks) ||
          totalChunks < 1 ||
          totalChunks > 64 ||
          chunkIndex < 0 ||
          chunkIndex >= totalChunks
        ) {
          return res.status(400).json({ error: "Invalid chunk index or count" });
        }
        if (!Number.isFinite(totalSize) || totalSize < 0 || totalSize > MAX_UPLOAD_BYTES) {
          return res.status(400).json({ error: "File exceeds 100 MB maximum limit" });
        }
        if (!req.file) {
          return res.status(400).json({ error: "Missing chunk payload" });
        }

        chunkDir = resolveInside(TMP_UPLOADS_DIR, uploadId);
        await fs.promises.mkdir(chunkDir, { recursive: true });
        const partPath = resolveInside(
          chunkDir,
          `part_${String(chunkIndex).padStart(4, "0")}`
        );
        await fs.promises.writeFile(partPath, req.file.buffer);

        if (chunkIndex < totalChunks - 1) {
          return res.json({ ok: true, complete: false, chunkIndex });
        }

        const catDir = resolveInside(IMPORT_DIR, category);
        await fs.promises.mkdir(catDir, { recursive: true });
        const finalSize = await assembleChunksAtomically(
          chunkDir,
          totalChunks,
          catDir,
          safeName
        );
        await fs.promises.rm(chunkDir, { recursive: true, force: true }).catch(() => {});

        return res.json({
          ok: true,
          complete: true,
          uploaded: { name: safeName, size: finalSize },
        });
      } catch (uploadErr) {
        if (chunkDir) {
          await fs.promises.rm(chunkDir, { recursive: true, force: true }).catch(() => {});
        }
        return res.status(400).json({ error: (uploadErr as Error).message });
      }
    });
  });

  app.delete("/api/import-container/:category/:filename", (req, res) => {
    try {
      const category = req.params.category;
      if (category !== "characters" && category !== "props") {
        return res.status(400).json({ error: "Invalid category" });
      }
      const filename = assertSafeSegment(req.params.filename, "filename");
      const target = resolveInside(IMPORT_DIR, category, filename);
      if (!fs.existsSync(target)) {
        return res.status(404).json({ error: "File not found" });
      }
      fs.unlinkSync(target);
      res.json({ deleted: filename });
    } catch (err) {
      res.status(400).json({ error: (err as Error).message });
    }
  });

  app.get("/api/import-container/preview/:filename", (req, res) => {
    try {
      const filename = assertSafeSegment(req.params.filename, "preview filename");
      if (!filename.toLowerCase().endsWith(".png")) {
        return res.status(400).json({ error: "Only PNG previews can be viewed" });
      }
      const target = resolveInside(PROFILES_DIR, filename);
      if (!fs.existsSync(target)) {
        return res.status(404).json({ error: "Preview not found" });
      }
      res.sendFile(target);
    } catch (err) {
      res.status(400).json({ error: (err as Error).message });
    }
  });

  function runCommandCapture(
    cmd: string,
    args: string[],
    envOverrides: NodeJS.ProcessEnv = {}
  ): Promise<{ code: number | null; logs: string }> {
    return new Promise((resolve, reject) => {
      const child = spawn(cmd, args, {
        cwd: ROOT_DIR,
        shell: false,
        env: { ...process.env, ...envOverrides },
      });
      let output = "";
      child.stdout.on("data", (chunk) => {
        output += chunk.toString();
      });
      child.stderr.on("data", (chunk) => {
        output += chunk.toString();
      });
      child.on("close", (code) => resolve({ code, logs: output }));
      child.on("error", (err) => reject(err));
    });
  }

  function resolveBlenderPath(): string | null {
    const candidates = [
      LOCAL_BLENDER_BIN,
      "/usr/local/bin/blender",
      "/opt/blender/blender",
      "/usr/bin/blender",
    ];
    for (const c of candidates) {
      if (fs.existsSync(c)) return c;
    }
    return null;
  }

  app.post("/api/profiler/run", async (req, res) => {
    if (profilerRunning) {
      return res.status(409).json({ error: "Profiler is already running" });
    }
    if (!fs.existsSync(PROFILER_SCRIPT)) {
      return res.status(500).json({ error: "scripts/profile_container.py not found" });
    }

    profilerRunning = true;
    let logs = "";
    try {
      const hasPil = fs.existsSync(path.join(PYDEPS_DIR, "PIL"));
      const hasXvfb =
        process.platform !== "linux" ||
        Boolean(process.env.DISPLAY) ||
        fs.existsSync("/usr/bin/xvfb-run");
      const hasLibGl =
        process.platform !== "linux" ||
        fs.existsSync("/usr/lib/x86_64-linux-gnu/libGL.so.1");
      const hasBlender = resolveBlenderPath() !== null;

      if ((!hasPil || !hasXvfb || !hasLibGl || !hasBlender) && fs.existsSync(BOOTSTRAP_SCRIPT)) {
        logs += "[bootstrap] Missing runtime prerequisites detected; running bootstrap_runtime.sh...\n";
        const boot = await runCommandCapture("bash", [BOOTSTRAP_SCRIPT]);
        logs += boot.logs;
        if (boot.code !== 0) {
          return res.status(500).json({
            ok: false,
            exitCode: boot.code,
            error: "Runtime prerequisite bootstrap failed",
            logs,
          });
        }
      }

      const force = req.body?.force === true;
      const pyArgs = [PROFILER_SCRIPT, "--container-dir", IMPORT_DIR];
      const blenderBin = resolveBlenderPath();
      if (blenderBin) {
        pyArgs.push("--blender", blenderBin);
      }
      if (force) {
        pyArgs.push("--force");
      }

      const needXvfb =
        process.platform === "linux" &&
        !process.env.DISPLAY &&
        fs.existsSync("/usr/bin/xvfb-run");

      const cmd = needXvfb ? "xvfb-run" : "python3";
      const cmdArgs = needXvfb ? ["-a", "python3", ...pyArgs] : pyArgs;

      const pyPathParts = [PYDEPS_DIR];
      if (process.env.PYTHONPATH) {
        pyPathParts.push(process.env.PYTHONPATH);
      }

      const result = await runCommandCapture(cmd, cmdArgs, {
        PYTHONUNBUFFERED: "1",
        PYTHONPATH: pyPathParts.join(path.delimiter),
        LP_NUM_THREADS: "1",
        OMP_NUM_THREADS: "2",
        MESA_GLTHREAD: "false",
      });
      logs += result.logs;

      return res.json({
        ok: result.code === 0,
        exitCode: result.code,
        logs,
      });
    } catch (err) {
      return res.status(500).json({
        ok: false,
        error: (err as Error).message,
        logs,
      });
    } finally {
      profilerRunning = false;
    }
  });

  function listDirFiles(dirPath: string) {
    if (!fs.existsSync(dirPath)) return [];
    return fs
      .readdirSync(dirPath, { withFileTypes: true })
      .filter((e) => e.isFile() && !e.name.startsWith("."))
      .sort((a, b) => a.name.localeCompare(b.name))
      .map((e) => {
        const st = fs.statSync(path.join(dirPath, e.name));
        return {
          name: e.name,
          size: st.size,
          mtimeMs: st.mtimeMs,
        };
      });
  }

  app.get("/api/batches", (_req, res) => {
    try {
      const batchEntries = fs
        .readdirSync(BATCHES_DIR, { withFileTypes: true })
        .filter((e) => e.isDirectory() && !e.name.startsWith("."))
        .sort((a, b) => a.name.localeCompare(b.name));

      const batches = batchEntries.map((b) => {
        const bDir = resolveInside(BATCHES_DIR, b.name);
        const epEntries = fs
          .readdirSync(bDir, { withFileTypes: true })
          .filter((e) => e.isDirectory() && !e.name.startsWith("."))
          .sort((a, b) => a.name.localeCompare(b.name));

        const episodes = epEntries.map((ep) => {
          const epDir = resolveInside(bDir, ep.name);
          return {
            id: ep.name,
            inputs: listDirFiles(path.join(epDir, "0_input")),
            outputs: listDirFiles(path.join(epDir, "5_output")),
          };
        });

        return { id: b.name, episodes };
      });

      res.json({ batches });
    } catch (err) {
      res.status(500).json({ error: (err as Error).message });
    }
  });

  app.post("/api/batches", (req, res) => {
    try {
      const batchId = assertSafeSegment(req.body?.batchId, "batchId");
      const target = resolveInside(BATCHES_DIR, batchId);
      if (fs.existsSync(target)) {
        return res.status(409).json({ error: "Batch already exists" });
      }
      fs.mkdirSync(target, { recursive: true });
      res.status(201).json({ id: batchId });
    } catch (err) {
      res.status(400).json({ error: (err as Error).message });
    }
  });

  app.delete("/api/batches/:batchId", (req, res) => {
    try {
      const batchId = assertSafeSegment(req.params.batchId, "batchId");
      const target = resolveInside(BATCHES_DIR, batchId);
      if (!fs.existsSync(target)) {
        return res.status(404).json({ error: "Batch not found" });
      }
      fs.rmSync(target, { recursive: true, force: true });
      res.json({ deleted: batchId });
    } catch (err) {
      res.status(400).json({ error: (err as Error).message });
    }
  });

  app.post("/api/batches/:batchId/episodes", (req, res) => {
    try {
      const batchId = assertSafeSegment(req.params.batchId, "batchId");
      const episodeId = assertSafeSegment(req.body?.episodeId, "episodeId");
      const batchDir = resolveInside(BATCHES_DIR, batchId);
      if (!fs.existsSync(batchDir)) {
        return res.status(404).json({ error: "Batch not found" });
      }
      const epDir = resolveInside(batchDir, episodeId);
      if (fs.existsSync(epDir)) {
        return res.status(409).json({ error: "Episode already exists" });
      }
      fs.mkdirSync(path.join(epDir, "0_input"), { recursive: true });
      fs.mkdirSync(path.join(epDir, "5_output"), { recursive: true });
      res.status(201).json({ id: episodeId });
    } catch (err) {
      res.status(400).json({ error: (err as Error).message });
    }
  });

  app.delete("/api/batches/:batchId/episodes/:episodeId", (req, res) => {
    try {
      const batchId = assertSafeSegment(req.params.batchId, "batchId");
      const episodeId = assertSafeSegment(req.params.episodeId, "episodeId");
      const epDir = resolveInside(BATCHES_DIR, batchId, episodeId);
      if (!fs.existsSync(epDir)) {
        return res.status(404).json({ error: "Episode not found" });
      }
      fs.rmSync(epDir, { recursive: true, force: true });
      res.json({ deleted: episodeId });
    } catch (err) {
      res.status(400).json({ error: (err as Error).message });
    }
  });

  const episodeInputStorage = multer.diskStorage({
    destination: (req, _file, cb) => {
      try {
        const batchId = assertSafeSegment(req.params.batchId, "batchId");
        const episodeId = assertSafeSegment(req.params.episodeId, "episodeId");
        const inputDir = resolveInside(BATCHES_DIR, batchId, episodeId, "0_input");
        fs.mkdirSync(inputDir, { recursive: true });
        cb(null, inputDir);
      } catch (err) {
        cb(err as Error, "");
      }
    },
    filename: (_req, file, cb) => {
      try {
        const ext = path.extname(file.originalname).toLowerCase();
        if (ext === ".json") {
          cb(null, "storyboard.json");
        } else if (ext === ".mp3") {
          cb(null, "narration.mp3");
        } else if (ext === ".wav") {
          cb(null, "narration.wav");
        } else {
          cb(new Error("Only .json, .mp3, or .wav allowed"), "");
        }
      } catch (err) {
        cb(err as Error, "");
      }
    },
  });

  const uploadEpisodeInput = multer({
    storage: episodeInputStorage,
    limits: { fileSize: 100 * 1024 * 1024 },
    fileFilter: (_req, file, cb) => {
      const ext = path.extname(file.originalname).toLowerCase();
      if (ext === ".json" || ext === ".mp3" || ext === ".wav") {
        cb(null, true);
      } else {
        cb(new Error("Only storyboard (.json) and narration (.mp3/.wav) files are allowed"));
      }
    },
  });

  app.post("/api/batches/:batchId/episodes/:episodeId/input", (req, res) => {
    uploadEpisodeInput.array("files", 5)(req, res, (err) => {
      if (err) {
        return res.status(400).json({ error: err.message });
      }
      const uploaded = ((req.files as Express.Multer.File[]) || []).map((f) => ({
        name: f.filename,
        size: f.size,
      }));
      res.json({ uploaded });
    });
  });

  app.post("/api/batches/:batchId/episodes/:episodeId/input/chunk", (req, res) => {
    uploadChunkMulter.single("chunk")(req, res, async (err) => {
      if (err) {
        return res.status(400).json({ error: err.message });
      }
      let chunkDir: string | null = null;
      try {
        const batchId = assertSafeSegment(req.params.batchId, "batchId");
        const episodeId = assertSafeSegment(req.params.episodeId, "episodeId");
        const uploadId = assertUploadId(req.body?.uploadId);
        const origName = path.basename(String(req.body?.filename || ""));
        const ext = path.extname(origName).toLowerCase();

        let targetFilename: string;
        if (ext === ".json") {
          targetFilename = "storyboard.json";
        } else if (ext === ".mp3") {
          targetFilename = "narration.mp3";
        } else if (ext === ".wav") {
          targetFilename = "narration.wav";
        } else {
          return res.status(400).json({
            error: "Only storyboard (.json) and narration (.mp3/.wav) files are allowed",
          });
        }

        const chunkIndex = Number(req.body?.chunkIndex);
        const totalChunks = Number(req.body?.totalChunks);
        const totalSize = Number(req.body?.totalSize);

        if (
          !Number.isInteger(chunkIndex) ||
          !Number.isInteger(totalChunks) ||
          totalChunks < 1 ||
          totalChunks > 64 ||
          chunkIndex < 0 ||
          chunkIndex >= totalChunks
        ) {
          return res.status(400).json({ error: "Invalid chunk index or count" });
        }
        if (!Number.isFinite(totalSize) || totalSize < 0 || totalSize > MAX_UPLOAD_BYTES) {
          return res.status(400).json({ error: "File exceeds 100 MB maximum limit" });
        }
        if (!req.file) {
          return res.status(400).json({ error: "Missing chunk payload" });
        }

        chunkDir = resolveInside(TMP_UPLOADS_DIR, uploadId);
        await fs.promises.mkdir(chunkDir, { recursive: true });
        const partPath = resolveInside(
          chunkDir,
          `part_${String(chunkIndex).padStart(4, "0")}`
        );
        await fs.promises.writeFile(partPath, req.file.buffer);

        if (chunkIndex < totalChunks - 1) {
          return res.json({ ok: true, complete: false, chunkIndex });
        }

        const inputDir = resolveInside(BATCHES_DIR, batchId, episodeId, "0_input");
        await fs.promises.mkdir(inputDir, { recursive: true });
        const finalSize = await assembleChunksAtomically(
          chunkDir,
          totalChunks,
          inputDir,
          targetFilename
        );
        await fs.promises.rm(chunkDir, { recursive: true, force: true }).catch(() => {});

        return res.json({
          ok: true,
          complete: true,
          uploaded: { name: targetFilename, size: finalSize },
        });
      } catch (uploadErr) {
        if (chunkDir) {
          await fs.promises.rm(chunkDir, { recursive: true, force: true }).catch(() => {});
        }
        return res.status(400).json({ error: (uploadErr as Error).message });
      }
    });
  });

  app.delete("/api/batches/:batchId/episodes/:episodeId/input/:filename", (req, res) => {
    try {
      const batchId = assertSafeSegment(req.params.batchId, "batchId");
      const episodeId = assertSafeSegment(req.params.episodeId, "episodeId");
      const filename = assertSafeSegment(req.params.filename, "filename");
      const target = resolveInside(BATCHES_DIR, batchId, episodeId, "0_input", filename);
      if (!fs.existsSync(target)) {
        return res.status(404).json({ error: "Input file not found" });
      }
      fs.unlinkSync(target);
      res.json({ deleted: filename });
    } catch (err) {
      res.status(400).json({ error: (err as Error).message });
    }
  });

  app.get("/api/batches/:batchId/episodes/:episodeId/output/:filename", (req, res) => {
    try {
      const batchId = assertSafeSegment(req.params.batchId, "batchId");
      const episodeId = assertSafeSegment(req.params.episodeId, "episodeId");
      const filename = assertSafeSegment(req.params.filename, "filename");
      const target = resolveInside(BATCHES_DIR, batchId, episodeId, "5_output", filename);
      if (!fs.existsSync(target)) {
        return res.status(404).json({ error: "Output file not found" });
      }
      res.download(target, filename);
    } catch (err) {
      res.status(400).json({ error: (err as Error).message });
    }
  });

  app.use("/api", (_req, res) => {
    res.status(404).json({ error: "API endpoint not found" });
  });

  app.use(
    (
      err: unknown,
      req: express.Request,
      res: express.Response,
      next: express.NextFunction
    ) => {
      if (res.headersSent) {
        return next(err);
      }
      const status =
        typeof (err as { status?: number })?.status === "number"
          ? (err as { status: number }).status
          : 500;
      const message = err instanceof Error ? err.message : "Internal server error";
      if (req.path.startsWith("/api")) {
        return res.status(status).json({ error: message });
      }
      return next(err);
    }
  );

  if (process.env.NODE_ENV !== "production") {
    const { createServer: createViteServer } = await import("vite");
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distDir = path.join(ROOT_DIR, "dist");
    app.use(express.static(distDir));
    app.get("*", (_req, res) => {
      res.sendFile(path.join(distDir, "index.html"));
    });
  }

  const port = Number(process.env.PORT) || 3000;
  const server = app.listen(port, "0.0.0.0", () => {
    console.log(`Persephone Workspace Manager listening on http://0.0.0.0:${port}`);
  });
  server.requestTimeout = 10 * 60 * 1000;
  server.keepAliveTimeout = 61 * 1000;
  server.headersTimeout = 65 * 1000;
}

startServer();
