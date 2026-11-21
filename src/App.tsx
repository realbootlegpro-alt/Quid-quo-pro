import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  Download,
  Eye,
  FileAudio,
  FileCode2,
  FileVideo,
  FolderPlus,
  Play,
  Plus,
  RefreshCw,
  Trash2,
  Upload,
  X,
} from "lucide-react";

type Category = "characters" | "props";

interface ImportFileItem {
  name: string;
  size: number;
  mtimeMs: number;
  status: "Profiled" | "Unprofiled";
  assetId: string | null;
  dimensions_m: number[] | null;
  previewFile: string | null;
}

interface BatchFileItem {
  name: string;
  size: number;
  mtimeMs: number;
}

interface EpisodeItem {
  id: string;
  inputs: BatchFileItem[];
  outputs: BatchFileItem[];
}

interface BatchItem {
  id: string;
  episodes: EpisodeItem[];
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

const CHUNK_SIZE_BYTES = 4 * 1024 * 1024;
const MAX_FILE_BYTES = 100 * 1024 * 1024;

function generateUploadId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID().replace(/-/g, "");
  }
  return `${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 12)}`;
}

async function apiFetchJson<T = Record<string, unknown>>(
  url: string,
  init?: RequestInit,
  retries = 1
): Promise<T> {
  let lastError: Error | null = null;
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const res = await fetch(url, {
        credentials: "same-origin",
        ...init,
      });

      if ((res.status === 502 || res.status === 503 || res.status === 504) && attempt < retries) {
        await new Promise((r) => setTimeout(r, 900));
        continue;
      }

      const rawText = await res.text();
      let parsed: unknown = null;
      const trimmed = rawText.trim();
      if (trimmed.startsWith("{") || trimmed.startsWith("[")) {
        try {
          parsed = JSON.parse(trimmed);
        } catch {
          parsed = null;
        }
      }

      if (parsed !== null && typeof parsed === "object") {
        const obj = parsed as Record<string, unknown>;
        if (!res.ok) {
          throw new Error(
            typeof obj.error === "string"
              ? obj.error
              : `Request failed (HTTP ${res.status})`
          );
        }
        return parsed as T;
      }

      const lower = trimmed.toLowerCase();
      if (lower.startsWith("<!doctype") || lower.startsWith("<html")) {
        if (lower.includes("starting server") || res.status === 502 || res.status === 503 || res.status === 504) {
          throw new Error("Server is restarting or warming up — please wait a moment and try again.");
        }
        if (lower.includes("cookie") || lower.includes("auth") || res.status === 401 || res.status === 403) {
          throw new Error("Session verification expired — please reload the page and try again.");
        }
        throw new Error(`Server returned an HTML page instead of JSON (HTTP ${res.status}). Try reloading the page.`);
      }

      if (!res.ok) {
        throw new Error(trimmed ? trimmed.slice(0, 200) : `Request failed (HTTP ${res.status})`);
      }

      throw new Error("Unexpected non-JSON response from server");
    } catch (err) {
      const e = err as Error;
      const isNetworkErr =
        e instanceof TypeError || /failed to fetch|networkerror|load failed/i.test(e.message);
      if (isNetworkErr && attempt < retries) {
        await new Promise((r) => setTimeout(r, 800));
        continue;
      }
      if (isNetworkErr) {
        lastError = new Error(
          "Upload/network request failed before reaching the server. Check your connection or reload the page if your session expired."
        );
      } else {
        lastError = e;
      }
      break;
    }
  }
  throw lastError || new Error("Request failed");
}

async function uploadFileInChunks(
  endpointUrl: string,
  file: File,
  onProgress?: (chunkIndex: number, totalChunks: number) => void
): Promise<void> {
  if (file.size > MAX_FILE_BYTES) {
    throw new Error(`${file.name} exceeds the 100 MB per-file limit.`);
  }
  const uploadId = generateUploadId();
  const totalChunks = Math.max(1, Math.ceil(file.size / CHUNK_SIZE_BYTES));

  for (let chunkIndex = 0; chunkIndex < totalChunks; chunkIndex++) {
    onProgress?.(chunkIndex, totalChunks);
    const start = chunkIndex * CHUNK_SIZE_BYTES;
    const end = Math.min(file.size, start + CHUNK_SIZE_BYTES);
    const slice = file.slice(start, end);

    const form = new FormData();
    form.append("uploadId", uploadId);
    form.append("filename", file.name);
    form.append("chunkIndex", String(chunkIndex));
    form.append("totalChunks", String(totalChunks));
    form.append("totalSize", String(file.size));
    form.append("chunk", slice, file.name);

    await apiFetchJson(
      endpointUrl,
      { method: "POST", body: form },
      1
    );
  }
}

export default function App() {
  const [category, setCategory] = useState<Category>("characters");
  const [importFiles, setImportFiles] = useState<ImportFileItem[]>([]);
  const [dragOver, setDragOver] = useState(false);
  const [uploadingAssets, setUploadingAssets] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<string | null>(null);
  const [runningProfiler, setRunningProfiler] = useState(false);
  const [profilerLogs, setProfilerLogs] = useState<string>("");
  const [previewModal, setPreviewModal] = useState<{
    name: string;
    url: string;
  } | null>(null);
  const [confirmDeleteAsset, setConfirmDeleteAsset] = useState<string | null>(null);
  const assetInputRef = useRef<HTMLInputElement | null>(null);

  const [batches, setBatches] = useState<BatchItem[]>([]);
  const [newBatchId, setNewBatchId] = useState("");
  const [newEpisodeByBatch, setNewEpisodeByBatch] = useState<Record<string, string>>({});
  const [confirmDeleteBatch, setConfirmDeleteBatch] = useState<string | null>(null);
  const [confirmDeleteEp, setConfirmDeleteEp] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const loadImportFiles = useCallback(async (cat: Category) => {
    const data = await apiFetchJson<{ files?: ImportFileItem[] }>(
      `/api/import-container/${cat}`
    );
    setImportFiles(data.files || []);
  }, []);

  const loadBatches = useCallback(async () => {
    const data = await apiFetchJson<{ batches?: BatchItem[] }>("/api/batches");
    setBatches(data.batches || []);
  }, []);

  const refreshAll = useCallback(async () => {
    try {
      setErrorMsg(null);
      await Promise.all([loadImportFiles(category), loadBatches()]);
    } catch (err) {
      setErrorMsg((err as Error).message);
    }
  }, [category, loadImportFiles, loadBatches]);

  useEffect(() => {
    refreshAll();
  }, [refreshAll]);

  const handleAssetUpload = async (fileList: FileList | null) => {
    if (!fileList || fileList.length === 0) return;
    const files = Array.from(fileList);
    for (const f of files) {
      const lower = f.name.toLowerCase();
      if (!lower.endsWith(".glb") && !lower.endsWith(".blend")) {
        setErrorMsg(`Unsupported file "${f.name}": only .glb and .blend files are allowed.`);
        if (assetInputRef.current) assetInputRef.current.value = "";
        return;
      }
      if (f.size > MAX_FILE_BYTES) {
        setErrorMsg(`"${f.name}" exceeds the 100 MB per-file limit.`);
        if (assetInputRef.current) assetInputRef.current.value = "";
        return;
      }
    }

    setErrorMsg(null);
    setUploadingAssets(true);
    try {
      for (let idx = 0; idx < files.length; idx++) {
        const file = files[idx];
        await uploadFileInChunks(
          `/api/import-container/${category}/upload/chunk`,
          file,
          (chunkIdx, totalChunks) => {
            const pct = Math.round(((chunkIdx + 1) / totalChunks) * 100);
            setUploadProgress(
              `Uploading ${file.name} (${idx + 1}/${files.length}) — ${pct}%`
            );
          }
        );
      }
      await loadImportFiles(category);
    } catch (err) {
      setErrorMsg((err as Error).message);
    } finally {
      setUploadingAssets(false);
      setUploadProgress(null);
      if (assetInputRef.current) assetInputRef.current.value = "";
    }
  };

  const handleDeleteAsset = async (filename: string) => {
    setErrorMsg(null);
    try {
      await apiFetchJson(
        `/api/import-container/${category}/${encodeURIComponent(filename)}`,
        { method: "DELETE" },
        0
      );
      setConfirmDeleteAsset(null);
      await loadImportFiles(category);
    } catch (err) {
      setErrorMsg((err as Error).message);
    }
  };

  const handleRunProfiler = async () => {
    setErrorMsg(null);
    setRunningProfiler(true);
    setProfilerLogs("Running scripts/profile_container.py...\n");
    try {
      const data = await apiFetchJson<{
        ok?: boolean;
        exitCode?: number;
        logs?: string;
        error?: string;
      }>(
        "/api/profiler/run",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ force: false }),
        },
        0
      );
      setProfilerLogs(data.logs || (data.error ? `Error: ${data.error}` : "Done."));
      if (!data.ok) {
        setErrorMsg(data.error || `Profiler exited with code ${data.exitCode}`);
      }
      await loadImportFiles(category);
    } catch (err) {
      setErrorMsg((err as Error).message);
      setProfilerLogs((prev) => prev + `\nError: ${(err as Error).message}`);
    } finally {
      setRunningProfiler(false);
    }
  };

  const handleCreateBatch = async (e: React.FormEvent) => {
    e.preventDefault();
    const id = newBatchId.trim();
    if (!id) return;
    setErrorMsg(null);
    try {
      await apiFetchJson(
        "/api/batches",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ batchId: id }),
        },
        0
      );
      setNewBatchId("");
      await loadBatches();
    } catch (err) {
      setErrorMsg((err as Error).message);
    }
  };

  const handleDeleteBatch = async (batchId: string) => {
    setErrorMsg(null);
    try {
      await apiFetchJson(
        `/api/batches/${encodeURIComponent(batchId)}`,
        { method: "DELETE" },
        0
      );
      setConfirmDeleteBatch(null);
      await loadBatches();
    } catch (err) {
      setErrorMsg((err as Error).message);
    }
  };

  const handleCreateEpisode = async (batchId: string) => {
    const epId = (newEpisodeByBatch[batchId] || "").trim();
    if (!epId) return;
    setErrorMsg(null);
    try {
      await apiFetchJson(
        `/api/batches/${encodeURIComponent(batchId)}/episodes`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ episodeId: epId }),
        },
        0
      );
      setNewEpisodeByBatch((prev) => ({ ...prev, [batchId]: "" }));
      await loadBatches();
    } catch (err) {
      setErrorMsg((err as Error).message);
    }
  };

  const handleDeleteEpisode = async (batchId: string, episodeId: string) => {
    setErrorMsg(null);
    try {
      await apiFetchJson(
        `/api/batches/${encodeURIComponent(batchId)}/episodes/${encodeURIComponent(
          episodeId
        )}`,
        { method: "DELETE" },
        0
      );
      setConfirmDeleteEp(null);
      await loadBatches();
    } catch (err) {
      setErrorMsg((err as Error).message);
    }
  };

  const handleUploadEpisodeInput = async (
    batchId: string,
    episodeId: string,
    fileList: FileList | null
  ) => {
    if (!fileList || fileList.length === 0) return;
    const files = Array.from(fileList);
    setErrorMsg(null);
    try {
      const endpoint = `/api/batches/${encodeURIComponent(batchId)}/episodes/${encodeURIComponent(
        episodeId
      )}/input/chunk`;
      for (const file of files) {
        await uploadFileInChunks(endpoint, file);
      }
      await loadBatches();
    } catch (err) {
      setErrorMsg((err as Error).message);
    }
  };

  const handleDeleteEpisodeInput = async (
    batchId: string,
    episodeId: string,
    filename: string
  ) => {
    setErrorMsg(null);
    try {
      await apiFetchJson(
        `/api/batches/${encodeURIComponent(batchId)}/episodes/${encodeURIComponent(
          episodeId
        )}/input/${encodeURIComponent(filename)}`,
        { method: "DELETE" },
        0
      );
      await loadBatches();
    } catch (err) {
      setErrorMsg((err as Error).message);
    }
  };

  return (
    <div className="min-h-screen bg-stone-100 text-stone-900 font-sans">
      <header className="border-b border-stone-300 bg-white px-6 py-3 flex items-center justify-between">
        <div>
          <h1 className="text-base font-semibold tracking-tight text-stone-900">
            Persephone Workspace &amp; Batch Manager
          </h1>
          <p className="text-xs text-stone-500 font-mono">
            import_container/ &middot; batches/ &middot; Pre-Prod profiler
          </p>
        </div>
        <button
          type="button"
          onClick={refreshAll}
          className="inline-flex items-center gap-1.5 rounded border border-stone-300 bg-white px-3 py-1.5 text-xs font-medium text-stone-700 hover:bg-stone-50"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          Refresh
        </button>
      </header>

      {errorMsg && (
        <div className="mx-6 mt-4 flex items-center justify-between rounded border border-red-300 bg-red-50 px-3 py-2 text-xs text-red-800">
          <span>{errorMsg}</span>
          <button
            type="button"
            onClick={() => setErrorMsg(null)}
            className="text-red-600 hover:text-red-900"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      )}

      <main className="p-6 grid grid-cols-1 lg:grid-cols-2 gap-6 items-start">
        <section className="rounded border border-stone-300 bg-white">
          <div className="border-b border-stone-200 px-4 py-3 flex flex-wrap items-center justify-between gap-3">
            <div>
              <h2 className="text-sm font-semibold text-stone-900">
                Panel 1 &mdash; Import Container
              </h2>
              <p className="text-xs font-mono text-stone-500">
                import_container/{category}/
              </p>
            </div>

            <div className="flex items-center gap-2">
              <div className="inline-flex rounded border border-stone-300 bg-stone-100 p-0.5 text-xs">
                <button
                  type="button"
                  onClick={() => {
                    setCategory("characters");
                    setConfirmDeleteAsset(null);
                  }}
                  className={`px-2.5 py-1 rounded-sm font-medium ${
                    category === "characters"
                      ? "bg-white text-stone-900 shadow-xs"
                      : "text-stone-600 hover:text-stone-900"
                  }`}
                >
                  characters/
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setCategory("props");
                    setConfirmDeleteAsset(null);
                  }}
                  className={`px-2.5 py-1 rounded-sm font-medium ${
                    category === "props"
                      ? "bg-white text-stone-900 shadow-xs"
                      : "text-stone-600 hover:text-stone-900"
                  }`}
                >
                  props/
                </button>
              </div>

              <button
                type="button"
                disabled={runningProfiler}
                onClick={handleRunProfiler}
                className="inline-flex items-center gap-1.5 rounded bg-stone-900 px-3 py-1.5 text-xs font-medium text-white hover:bg-stone-800 disabled:opacity-50"
              >
                <Play className="w-3.5 h-3.5" />
                {runningProfiler ? "Profiling..." : "Run Profiler"}
              </button>
            </div>
          </div>

          <div className="p-4 space-y-4">
            <div
              onDragOver={(e) => {
                e.preventDefault();
                setDragOver(true);
              }}
              onDragLeave={() => setDragOver(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragOver(false);
                handleAssetUpload(e.dataTransfer.files);
              }}
              onClick={() => assetInputRef.current?.click()}
              className={`cursor-pointer rounded border border-dashed px-4 py-5 text-center transition-colors ${
                dragOver
                  ? "border-stone-900 bg-stone-100"
                  : "border-stone-300 bg-stone-50 hover:border-stone-400"
              }`}
            >
              <input
                ref={assetInputRef}
                type="file"
                multiple
                accept=".glb,.blend"
                className="hidden"
                onChange={(e) => handleAssetUpload(e.target.files)}
              />
              <Upload className="w-5 h-5 mx-auto mb-1.5 text-stone-500" />
              <p className="text-xs font-medium text-stone-800">
                {uploadingAssets
                  ? uploadProgress || "Uploading into persistent storage..."
                  : `Drop .glb or .blend files into import_container/${category}/`}
              </p>
              <p className="text-[11px] text-stone-500 mt-0.5">
                Chunked multi-file upload supported &middot; Max 100 MB per file
              </p>
            </div>

            <div className="border border-stone-200 rounded overflow-hidden">
              <table className="w-full text-left text-xs">
                <thead className="bg-stone-50 border-b border-stone-200 text-stone-600">
                  <tr>
                    <th className="py-2 px-3 font-medium">File</th>
                    <th className="py-2 px-3 font-medium">Size</th>
                    <th className="py-2 px-3 font-medium">Status</th>
                    <th className="py-2 px-3 font-medium text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-stone-200">
                  {importFiles.length === 0 ? (
                    <tr>
                      <td
                        colSpan={4}
                        className="py-6 px-3 text-center text-stone-400"
                      >
                        No .glb or .blend files in import_container/{category}/
                      </td>
                    </tr>
                  ) : (
                    importFiles.map((file) => (
                      <tr key={file.name} className="hover:bg-stone-50">
                        <td className="py-2 px-3 font-mono text-stone-900 break-all">
                          {file.name}
                          {file.dimensions_m && (
                            <span className="block text-[11px] text-stone-500">
                              {file.dimensions_m.map((d) => `${d}m`).join(" x ")}
                            </span>
                          )}
                        </td>
                        <td className="py-2 px-3 text-stone-600 whitespace-nowrap">
                          {formatBytes(file.size)}
                        </td>
                        <td className="py-2 px-3 whitespace-nowrap">
                          <span
                            className={`inline-block rounded px-2 py-0.5 text-[11px] font-medium ${
                              file.status === "Profiled"
                                ? "bg-emerald-50 text-emerald-700 border border-emerald-200"
                                : "bg-amber-50 text-amber-700 border border-amber-200"
                            }`}
                          >
                            {file.status}
                          </span>
                        </td>
                        <td className="py-2 px-3 text-right whitespace-nowrap space-x-1.5">
                          {file.previewFile && (
                            <button
                              type="button"
                              onClick={() =>
                                setPreviewModal({
                                  name: file.name,
                                  url: `/api/import-container/preview/${encodeURIComponent(
                                    file.previewFile!
                                  )}?t=${Date.now()}`,
                                })
                              }
                              className="inline-flex items-center gap-1 rounded border border-stone-300 bg-white px-2 py-1 text-[11px] text-stone-700 hover:bg-stone-100"
                            >
                              <Eye className="w-3 h-3" />
                              Preview
                            </button>
                          )}
                          {confirmDeleteAsset === file.name ? (
                            <>
                              <button
                                type="button"
                                onClick={() => handleDeleteAsset(file.name)}
                                className="inline-flex items-center rounded bg-red-600 px-2 py-1 text-[11px] font-medium text-white hover:bg-red-700"
                              >
                                Confirm
                              </button>
                              <button
                                type="button"
                                onClick={() => setConfirmDeleteAsset(null)}
                                className="inline-flex items-center rounded border border-stone-300 bg-white px-2 py-1 text-[11px] text-stone-600 hover:bg-stone-100"
                              >
                                Cancel
                              </button>
                            </>
                          ) : (
                            <button
                              type="button"
                              onClick={() => setConfirmDeleteAsset(file.name)}
                              className="inline-flex items-center gap-1 rounded border border-stone-300 bg-white px-2 py-1 text-[11px] text-red-600 hover:bg-red-50"
                            >
                              <Trash2 className="w-3 h-3" />
                              Delete
                            </button>
                          )}
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>

            {profilerLogs && (
              <div>
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs font-medium text-stone-700">
                    Profiler Output
                  </span>
                  <button
                    type="button"
                    onClick={() => setProfilerLogs("")}
                    className="text-[11px] text-stone-500 hover:text-stone-800"
                  >
                    Clear
                  </button>
                </div>
                <pre className="max-h-56 overflow-auto rounded border border-stone-300 bg-stone-950 p-3 text-[11px] font-mono text-stone-100 whitespace-pre-wrap">
                  {profilerLogs}
                </pre>
              </div>
            )}
          </div>
        </section>

        <section className="rounded border border-stone-300 bg-white">
          <div className="border-b border-stone-200 px-4 py-3 flex flex-wrap items-center justify-between gap-3">
            <div>
              <h2 className="text-sm font-semibold text-stone-900">
                Panel 2 &mdash; Batches
              </h2>
              <p className="text-xs font-mono text-stone-500">
                batches/&lt;batch_id&gt;/&lt;ep_id&gt;/&#123;0_input, 5_output&#125;
              </p>
            </div>

            <form onSubmit={handleCreateBatch} className="flex items-center gap-2">
              <input
                type="text"
                value={newBatchId}
                onChange={(e) => setNewBatchId(e.target.value)}
                placeholder="batch_001"
                className="w-36 rounded border border-stone-300 px-2.5 py-1 text-xs font-mono focus:border-stone-900 focus:outline-none"
              />
              <button
                type="submit"
                className="inline-flex items-center gap-1 rounded bg-stone-900 px-3 py-1.5 text-xs font-medium text-white hover:bg-stone-800"
              >
                <FolderPlus className="w-3.5 h-3.5" />
                New Batch
              </button>
            </form>
          </div>

          <div className="p-4 space-y-4">
            {batches.length === 0 ? (
              <div className="rounded border border-stone-200 bg-stone-50 py-8 px-4 text-center text-xs text-stone-500">
                No batch folders yet. Create e.g. <span className="font-mono">batch_001</span> above.
              </div>
            ) : (
              batches.map((batch) => (
                <div
                  key={batch.id}
                  className="rounded border border-stone-300 bg-stone-50 overflow-hidden"
                >
                  <div className="border-b border-stone-200 bg-white px-3 py-2 flex flex-wrap items-center justify-between gap-2">
                    <span className="text-xs font-mono font-semibold text-stone-900">
                      batches/{batch.id}/
                    </span>
                    <div className="flex items-center gap-2">
                      <input
                        type="text"
                        value={newEpisodeByBatch[batch.id] || ""}
                        onChange={(e) =>
                          setNewEpisodeByBatch((prev) => ({
                            ...prev,
                            [batch.id]: e.target.value,
                          }))
                        }
                        placeholder="ep_01_souffle"
                        className="w-36 rounded border border-stone-300 px-2 py-1 text-xs font-mono focus:border-stone-900 focus:outline-none"
                      />
                      <button
                        type="button"
                        onClick={() => handleCreateEpisode(batch.id)}
                        className="inline-flex items-center gap-1 rounded border border-stone-300 bg-white px-2.5 py-1 text-xs font-medium text-stone-800 hover:bg-stone-100"
                      >
                        <Plus className="w-3 h-3" />
                        Add Episode
                      </button>

                      {confirmDeleteBatch === batch.id ? (
                        <>
                          <button
                            type="button"
                            onClick={() => handleDeleteBatch(batch.id)}
                            className="rounded bg-red-600 px-2 py-1 text-[11px] font-medium text-white hover:bg-red-700"
                          >
                            Confirm
                          </button>
                          <button
                            type="button"
                            onClick={() => setConfirmDeleteBatch(null)}
                            className="rounded border border-stone-300 bg-white px-2 py-1 text-[11px] text-stone-600"
                          >
                            Cancel
                          </button>
                        </>
                      ) : (
                        <button
                          type="button"
                          onClick={() => setConfirmDeleteBatch(batch.id)}
                          className="inline-flex items-center gap-1 rounded border border-stone-300 bg-white px-2 py-1 text-[11px] text-red-600 hover:bg-red-50"
                        >
                          <Trash2 className="w-3 h-3" />
                          Delete Batch
                        </button>
                      )}
                    </div>
                  </div>

                  <div className="p-3 space-y-3">
                    {batch.episodes.length === 0 ? (
                      <p className="text-xs text-stone-500 px-1">
                        No episodes in this batch (max 3 episodes per Persephone batch).
                      </p>
                    ) : (
                      batch.episodes.map((ep) => {
                        const epKey = `${batch.id}/${ep.id}`;
                        return (
                          <div
                            key={ep.id}
                            className="rounded border border-stone-200 bg-white p-3 space-y-3"
                          >
                            <div className="flex items-center justify-between">
                              <span className="text-xs font-mono font-semibold text-stone-900">
                                {ep.id}
                              </span>
                              <div className="flex items-center gap-1.5">
                                {confirmDeleteEp === epKey ? (
                                  <>
                                    <button
                                      type="button"
                                      onClick={() =>
                                        handleDeleteEpisode(batch.id, ep.id)
                                      }
                                      className="rounded bg-red-600 px-2 py-0.5 text-[11px] font-medium text-white"
                                    >
                                      Confirm
                                    </button>
                                    <button
                                      type="button"
                                      onClick={() => setConfirmDeleteEp(null)}
                                      className="rounded border border-stone-300 px-2 py-0.5 text-[11px] text-stone-600"
                                    >
                                      Cancel
                                    </button>
                                  </>
                                ) : (
                                  <button
                                    type="button"
                                    onClick={() => setConfirmDeleteEp(epKey)}
                                    className="inline-flex items-center gap-1 rounded border border-stone-200 px-2 py-0.5 text-[11px] text-red-600 hover:bg-red-50"
                                  >
                                    <Trash2 className="w-3 h-3" />
                                    Delete
                                  </button>
                                )}
                              </div>
                            </div>

                            <div className="border-t border-stone-100 pt-2">
                              <div className="flex items-center justify-between mb-1.5">
                                <span className="text-[11px] font-mono font-medium text-stone-600">
                                  0_input/ (storyboard.json, narration.mp3/.wav)
                                </span>
                                <label className="cursor-pointer inline-flex items-center gap-1 rounded border border-stone-300 bg-stone-50 px-2 py-1 text-[11px] font-medium text-stone-700 hover:bg-stone-100">
                                  <Upload className="w-3 h-3" />
                                  Upload / Replace
                                  <input
                                    type="file"
                                    multiple
                                    accept=".json,.mp3,.wav"
                                    className="hidden"
                                    onChange={(e) => {
                                      handleUploadEpisodeInput(
                                        batch.id,
                                        ep.id,
                                        e.target.files
                                      );
                                      e.target.value = "";
                                    }}
                                  />
                                </label>
                              </div>

                              {ep.inputs.length === 0 ? (
                                <p className="text-[11px] text-stone-400">
                                  No input files uploaded yet.
                                </p>
                              ) : (
                                <ul className="space-y-1">
                                  {ep.inputs.map((inp) => (
                                    <li
                                      key={inp.name}
                                      className="flex items-center justify-between rounded bg-stone-50 px-2 py-1 text-xs"
                                    >
                                      <span className="inline-flex items-center gap-1.5 font-mono text-stone-800">
                                        {inp.name.endsWith(".json") ? (
                                          <FileCode2 className="w-3.5 h-3.5 text-stone-500" />
                                        ) : (
                                          <FileAudio className="w-3.5 h-3.5 text-stone-500" />
                                        )}
                                        {inp.name}
                                        <span className="text-stone-400">
                                          ({formatBytes(inp.size)})
                                        </span>
                                      </span>
                                      <button
                                        type="button"
                                        onClick={() =>
                                          handleDeleteEpisodeInput(
                                            batch.id,
                                            ep.id,
                                            inp.name
                                          )
                                        }
                                        className="text-[11px] text-red-600 hover:underline"
                                      >
                                        Delete
                                      </button>
                                    </li>
                                  ))}
                                </ul>
                              )}
                            </div>

                            <div className="border-t border-stone-100 pt-2">
                              <span className="block text-[11px] font-mono font-medium text-stone-600 mb-1.5">
                                5_output/ (_master.mp4, .srt, _cues.txt)
                              </span>
                              {ep.outputs.length === 0 ? (
                                <p className="text-[11px] text-stone-400">
                                  No outputs generated yet.
                                </p>
                              ) : (
                                <ul className="space-y-1">
                                  {ep.outputs.map((out) => (
                                    <li
                                      key={out.name}
                                      className="flex items-center justify-between rounded bg-stone-50 px-2 py-1 text-xs"
                                    >
                                      <span className="inline-flex items-center gap-1.5 font-mono text-stone-800">
                                        <FileVideo className="w-3.5 h-3.5 text-stone-500" />
                                        {out.name}
                                        <span className="text-stone-400">
                                          ({formatBytes(out.size)})
                                        </span>
                                      </span>
                                      <a
                                        href={`/api/batches/${encodeURIComponent(
                                          batch.id
                                        )}/episodes/${encodeURIComponent(
                                          ep.id
                                        )}/output/${encodeURIComponent(out.name)}`}
                                        download={out.name}
                                        className="inline-flex items-center gap-1 rounded border border-stone-300 bg-white px-2 py-0.5 text-[11px] font-medium text-stone-800 hover:bg-stone-100"
                                      >
                                        <Download className="w-3 h-3" />
                                        Download
                                      </a>
                                    </li>
                                  ))}
                                </ul>
                              )}
                            </div>
                          </div>
                        );
                      })
                    )}
                  </div>
                </div>
              ))
            )}
          </div>
        </section>
      </main>

      {previewModal && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4"
          onClick={() => setPreviewModal(null)}
        >
          <div
            className="max-w-3xl w-full rounded border border-stone-700 bg-stone-900 text-white overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between border-b border-stone-800 px-4 py-2.5">
              <span className="text-xs font-mono">{previewModal.name}</span>
              <button
                type="button"
                onClick={() => setPreviewModal(null)}
                className="text-stone-400 hover:text-white"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
            <div className="p-4 flex justify-center bg-stone-950">
              <img
                src={previewModal.url}
                alt={previewModal.name}
                className="max-h-[75vh] w-auto object-contain"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
