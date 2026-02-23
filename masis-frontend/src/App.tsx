import { useEffect, useState, useRef } from "react"
import { api } from "./api/temp"
import type { Document } from "./types"
import axios from "axios"

// ─── Types ───────────────────────────────────────────────────────────────────

interface MASISResult {
  status: "success" | "needs_clarification"
  answer: string | null
  confidence: number
  requires_human_review: boolean
  clarification_question: string | null
  critique: {
    hallucination_detected: boolean
    unsupported_claims: string[]
    logical_gaps: string[]
    conflicting_evidence: string[]
    needs_retry: boolean
  } | null
  evaluation: {
    faithfulness: number
    relevance: number
    completeness: number
    reasoning_quality: number
    overall_score: number
    improvement_suggestions: string[]
  } | null
  trace: Array<{
    node: string
    decision?: string
    confidence?: number
    retry_count?: number
    chunks?: number
    avg_score?: number
    duration_ms?: number
    warning?: string
    [key: string]: any
  }>
  metrics: {
    avg_retrieval_score: number
    retrieval_scores: number[]
    citation_count: number
    answer_length: number
    confidence_history: number[]
    node_latency_ms: Record<string, number>
    citation_violations: any[]
    evaluation: any
    [key: string]: any
  }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

const confidenceMeta = (v: number) => {
  if (v >= 0.85) return { label: "HIGH", color: "#22c55e", bg: "rgba(34,197,94,0.1)" }
  if (v >= 0.70) return { label: "MEDIUM", color: "#f59e0b", bg: "rgba(245,158,11,0.1)" }
  return { label: "LOW", color: "#ef4444", bg: "rgba(239,68,68,0.1)" }
}

const NODE_COLORS: Record<string, string> = {
  supervisor:   "#6366f1",
  researcher:   "#0ea5e9",
  synthesizer:  "#8b5cf6",
  critic:       "#f59e0b",
  evaluator:    "#22c55e",
}

// ─── Sub-components ───────────────────────────────────────────────────────────

function ScoreBar({ label, value, delay = 0 }: { label: string; value: number; delay?: number }) {
  const [width, setWidth] = useState(0)
  const pct = Math.round(value * 100)

  useEffect(() => {
    const t = setTimeout(() => setWidth(pct), delay + 100)
    return () => clearTimeout(t)
  }, [pct, delay])

  const color = pct >= 85 ? "#22c55e" : pct >= 70 ? "#f59e0b" : "#ef4444"

  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
        <span style={{ fontSize: 12, color: "#94a3b8", letterSpacing: "0.08em", textTransform: "uppercase" }}>
          {label}
        </span>
        <span style={{ fontSize: 13, fontFamily: "'JetBrains Mono', monospace", color, fontWeight: 600 }}>
          {pct}%
        </span>
      </div>
      <div style={{ height: 4, background: "rgba(255,255,255,0.06)", borderRadius: 2, overflow: "hidden" }}>
        <div style={{
          height: "100%",
          width: `${width}%`,
          background: color,
          borderRadius: 2,
          transition: "width 0.8s cubic-bezier(0.16, 1, 0.3, 1)",
          boxShadow: `0 0 8px ${color}80`,
        }} />
      </div>
    </div>
  )
}

function TraceEntry({ entry, index }: { entry: any; index: number }) {
  const color = NODE_COLORS[entry.node] || "#64748b"
  const isDecision = !!entry.decision

  return (
    <div style={{
      display: "flex",
      gap: 12,
      animation: `fadeSlideIn 0.3s ease both`,
      animationDelay: `${index * 0.05}s`,
    }}>
      {/* Timeline spine */}
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", width: 16 }}>
        <div style={{
          width: 10, height: 10, borderRadius: "50%",
          background: color,
          border: `2px solid ${color}40`,
          boxShadow: `0 0 6px ${color}60`,
          flexShrink: 0, marginTop: 3,
        }} />
        <div style={{ flex: 1, width: 1, background: "rgba(255,255,255,0.06)", marginTop: 4 }} />
      </div>

      {/* Content */}
      <div style={{ paddingBottom: 16, flex: 1 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
          <span style={{
            fontSize: 10, fontFamily: "'JetBrains Mono', monospace",
            color, letterSpacing: "0.1em", textTransform: "uppercase", fontWeight: 700,
          }}>
            {entry.node}
          </span>
          {isDecision && (
            <span style={{
              fontSize: 9, padding: "1px 6px", borderRadius: 3,
              background: entry.decision === "finalize" ? "rgba(34,197,94,0.15)"
                : entry.decision?.startsWith("HITL") ? "rgba(239,68,68,0.15)"
                : "rgba(99,102,241,0.15)",
              color: entry.decision === "finalize" ? "#22c55e"
                : entry.decision?.startsWith("HITL") ? "#ef4444"
                : "#818cf8",
              letterSpacing: "0.08em", textTransform: "uppercase", fontWeight: 600,
            }}>
              {entry.decision}
            </span>
          )}
          {entry.warning && (
            <span style={{
              fontSize: 9, padding: "1px 6px", borderRadius: 3,
              background: "rgba(239,68,68,0.15)", color: "#ef4444",
              letterSpacing: "0.08em", textTransform: "uppercase", fontWeight: 600,
            }}>
              ⚠ {entry.warning}
            </span>
          )}
          {entry.duration_ms !== undefined && (
            <span style={{ marginLeft: "auto", fontSize: 10, color: "#475569", fontFamily: "'JetBrains Mono', monospace" }}>
              {entry.duration_ms}ms
            </span>
          )}
        </div>

        <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
          {entry.chunks !== undefined && (
            <Chip label="chunks" value={entry.chunks} />
          )}
          {entry.avg_score !== undefined && (
            <Chip label="avg score" value={entry.avg_score} />
          )}
          {entry.confidence !== undefined && (
            <Chip label="confidence" value={`${(entry.confidence * 100).toFixed(0)}%`} />
          )}
          {entry.citations !== undefined && (
            <Chip label="citations" value={entry.citations} />
          )}
          {entry.invalid_citations !== undefined && entry.invalid_citations > 0 && (
            <Chip label="invalid cites" value={entry.invalid_citations} danger />
          )}
          {entry.hallucination !== undefined && (
            <Chip label="hallucination" value={entry.hallucination ? "YES" : "NO"} danger={entry.hallucination} />
          )}
          {entry.overall_score !== undefined && (
            <Chip label="overall" value={`${(entry.overall_score * 100).toFixed(0)}%`} />
          )}
          {entry.reason && (
            <Chip label="reason" value={entry.reason} />
          )}
          {entry.augmented_query_used !== undefined && entry.augmented_query_used && (
            <Chip label="augmented query" value="YES" />
          )}
          {entry.context_compressed !== undefined && entry.context_compressed && (
            <Chip label="compressed" value="YES" />
          )}
        </div>
      </div>
    </div>
  )
}

function Chip({ label, value, danger }: { label: string; value: any; danger?: boolean }) {
  return (
    <div style={{
      display: "inline-flex", alignItems: "center", gap: 4,
      padding: "2px 8px", borderRadius: 4,
      background: danger ? "rgba(239,68,68,0.1)" : "rgba(255,255,255,0.04)",
      border: `1px solid ${danger ? "rgba(239,68,68,0.2)" : "rgba(255,255,255,0.08)"}`,
    }}>
      <span style={{ fontSize: 10, color: "#475569", letterSpacing: "0.06em" }}>{label}</span>
      <span style={{
        fontSize: 10, fontFamily: "'JetBrains Mono', monospace",
        color: danger ? "#ef4444" : "#94a3b8", fontWeight: 600,
      }}>
        {String(value)}
      </span>
    </div>
  )
}

function ConfidenceRing({ value }: { value: number }) {
  const meta = confidenceMeta(value)
  const r = 36
  const circ = 2 * Math.PI * r
  const [dash, setDash] = useState(0)

  useEffect(() => {
    const t = setTimeout(() => setDash(value * circ), 200)
    return () => clearTimeout(t)
  }, [value, circ])

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
      <svg width={90} height={90} style={{ transform: "rotate(-90deg)" }}>
        <circle cx={45} cy={45} r={r} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth={6} />
        <circle
          cx={45} cy={45} r={r} fill="none"
          stroke={meta.color}
          strokeWidth={6}
          strokeDasharray={circ}
          strokeDashoffset={circ - dash}
          strokeLinecap="round"
          style={{ transition: "stroke-dashoffset 1s cubic-bezier(0.16, 1, 0.3, 1)", filter: `drop-shadow(0 0 6px ${meta.color}80)` }}
        />
      </svg>
      <div>
        <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 28, fontWeight: 700, color: meta.color, lineHeight: 1 }}>
          {(value * 100).toFixed(1)}%
        </div>
        <div style={{ fontSize: 10, letterSpacing: "0.15em", color: "#475569", textTransform: "uppercase", marginTop: 4 }}>
          {meta.label} CONFIDENCE
        </div>
      </div>
    </div>
  )
}

// ─── Main App ─────────────────────────────────────────────────────────────────

function App() {
  const [workspaces, setWorkspaces] = useState<string[]>([])
  const [selectedWorkspace, setSelectedWorkspace] = useState<string>("")
  const [documents, setDocuments] = useState<Document[]>([])
  const [files, setFiles] = useState<FileList | null>(null)
  const [newWorkspace, setNewWorkspace] = useState("")
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [query, setQuery] = useState("")
  const [masisResult, setMasisResult] = useState<MASISResult | null>(null)
  const [masisLoading, setMasisLoading] = useState(false)
  const [masisError, setMasisError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<"answer" | "trace" | "metrics">("answer")
  const [wsDropdownOpen, setWsDropdownOpen] = useState(false)

  const resultRef = useRef<HTMLDivElement>(null)
  const wsDropdownRef = useRef<HTMLDivElement>(null)

  // ─── Data fetching ──────────────────────────────────────────────────────────

  const fetchWorkspaces = async () => {
    try {
      const res = await api.get("/workspaces")
      setWorkspaces(res.data)
      if (!selectedWorkspace && res.data.length > 0) setSelectedWorkspace(res.data[0])
    } catch {
      setError("Failed to load workspaces")
    }
  }

  const fetchDocuments = async (workspace: string) => {
    try {
      const res = await api.get(`/workspaces/${workspace}/documents`)
      setDocuments(res.data)
    } catch {
      setError("Failed to load documents")
    }
  }

  useEffect(() => { fetchWorkspaces() }, [])
  useEffect(() => { if (selectedWorkspace) fetchDocuments(selectedWorkspace) }, [selectedWorkspace])

  // Close workspace dropdown when clicking outside
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (wsDropdownRef.current && !wsDropdownRef.current.contains(e.target as Node)) {
        setWsDropdownOpen(false)
      }
    }
    document.addEventListener("mousedown", handler)
    return () => document.removeEventListener("mousedown", handler)
  }, [])

  // ─── Actions ────────────────────────────────────────────────────────────────

  const handleCreateWorkspace = async () => {
    if (!newWorkspace.trim()) return setError("Workspace name cannot be empty")
    try {
      await api.post(`/workspaces/${newWorkspace}`)
      setSelectedWorkspace(newWorkspace)
      setNewWorkspace("")
      fetchWorkspaces()
    } catch (err) {
      setError(axios.isAxiosError(err) && err.response?.status === 409
        ? "Workspace already exists"
        : "Failed to create workspace")
    }
  }

  const handleDeleteWorkspace = async () => {
    try {
      await api.delete(`/workspaces/${selectedWorkspace}`)
      setSelectedWorkspace("")
      setDocuments([])
      fetchWorkspaces()
    } catch { setError("Failed to delete workspace") }
  }

  const handleUpload = async () => {
    if (!files || !selectedWorkspace) return
    setLoading(true)
    for (const file of Array.from(files)) {
      const formData = new FormData()
      formData.append("file", file)
      try {
        await api.post(`/workspaces/${selectedWorkspace}/upload`, formData)
      } catch { setError(`Upload failed: ${file.name}`) }
    }
    setFiles(null)
    fetchDocuments(selectedWorkspace)
    setLoading(false)
  }

  const handleDeleteDocument = async (docId: string, fileName: string) => {
    if (!confirm(`Delete "${fileName}"?`)) return
    try {
      await api.delete(`/workspaces/${selectedWorkspace}/documents/${docId}`)
      fetchDocuments(selectedWorkspace)
    } catch {
      setError(`Failed to delete ${fileName}`)
    }
  }

  const handleMasisQuery = async () => {
    if (!selectedWorkspace || !query.trim()) return setMasisError("Select a workspace and enter a query")
    setMasisLoading(true)
    setMasisError(null)
    setMasisResult(null)
    try {
      const res = await api.post(`/masis/workspaces/${selectedWorkspace}`, { query })
      setMasisResult(res.data)
      setActiveTab("answer")
      setTimeout(() => resultRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 100)
    } catch { setMasisError("Query failed — check server logs") }
    setMasisLoading(false)
  }

  const retryCount = masisResult?.trace?.filter(t => t.node === "supervisor" && t.decision === "retry").length ?? 0

  // ─── Render ─────────────────────────────────────────────────────────────────

  return (
    <>
      <style>{`
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

        body {
          background: #080c14;
          color: #e2e8f0;
          font-family: 'Syne', sans-serif;
          min-height: 100vh;
        }

        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 2px; }

        ::placeholder { color: #334155; }

        @keyframes fadeSlideIn {
          from { opacity: 0; transform: translateY(6px); }
          to   { opacity: 1; transform: translateY(0); }
        }

        @keyframes pulse-ring {
          0%, 100% { opacity: 0.6; }
          50%       { opacity: 1; }
        }

        @keyframes spin {
          to { transform: rotate(360deg); }
        }

        .sidebar-doc:hover { background: rgba(255,255,255,0.06) !important; }
        .sidebar-doc:hover .doc-delete-btn { opacity: 1 !important; }

        .ws-option:hover { background: rgba(99,102,241,0.1); }

        .tab-btn { cursor: pointer; transition: all 0.2s; }
        .tab-btn:hover { color: #e2e8f0 !important; }

        .action-btn {
          cursor: pointer;
          transition: all 0.2s;
          outline: none;
          border: none;
        }
        .action-btn:hover:not(:disabled) { filter: brightness(1.15); transform: translateY(-1px); }
        .action-btn:active:not(:disabled) { transform: translateY(0); }
        .action-btn:disabled { opacity: 0.4; cursor: not-allowed; }

        input, textarea, select {
          outline: none;
          font-family: 'Syne', sans-serif;
        }
        input:focus, textarea:focus, select:focus {
          border-color: rgba(99,102,241,0.6) !important;
          box-shadow: 0 0 0 3px rgba(99,102,241,0.1);
        }
      `}</style>

      <div style={{ display: "flex", height: "100vh", overflow: "hidden" }}>

        {/* ── SIDEBAR ─────────────────────────────────────────────────────── */}
        <aside style={{
          width: 280,
          background: "rgba(255,255,255,0.02)",
          borderRight: "1px solid rgba(255,255,255,0.06)",
          display: "flex",
          flexDirection: "column",
          flexShrink: 0,
        }}>

          {/* Logo */}
          <div style={{ padding: "28px 24px 20px", borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <div style={{
                width: 32, height: 32, borderRadius: 8,
                background: "linear-gradient(135deg, #6366f1, #8b5cf6)",
                display: "flex", alignItems: "center", justifyContent: "center",
                fontSize: 14, fontWeight: 800,
              }}>M</div>
              <div>
                <div style={{ fontWeight: 800, fontSize: 15, letterSpacing: "0.05em" }}>MASIS</div>
                <div style={{ fontSize: 9, color: "#475569", letterSpacing: "0.12em", textTransform: "uppercase" }}>
                  Strategic Intelligence
                </div>
              </div>
            </div>
          </div>

          {/* Workspace selector */}
          <div style={{ padding: "20px 20px 0" }}>
            <div style={{ fontSize: 10, color: "#475569", letterSpacing: "0.12em", textTransform: "uppercase", marginBottom: 8 }}>
              Workspace
            </div>

            {/* Custom styled dropdown — replaces native <select> for full theme control */}
            <div ref={wsDropdownRef} style={{ position: "relative" }}>
              <button
                className="action-btn"
                onClick={() => setWsDropdownOpen(o => !o)}
                style={{
                  width: "100%", padding: "10px 14px",
                  background: "rgba(255,255,255,0.05)",
                  border: "1px solid rgba(255,255,255,0.1)",
                  borderRadius: 8, color: selectedWorkspace ? "#e2e8f0" : "#475569",
                  fontSize: 13, textAlign: "left",
                  display: "flex", alignItems: "center", justifyContent: "space-between",
                  transition: "border-color 0.2s, background 0.2s",
                }}
              >
                <span>{selectedWorkspace || "— select workspace —"}</span>
                <svg
                  width="12" height="12" viewBox="0 0 12 12" fill="none"
                  style={{ transform: wsDropdownOpen ? "rotate(180deg)" : "rotate(0deg)", transition: "transform 0.2s", flexShrink: 0 }}
                >
                  <path d="M2 4l4 4 4-4" stroke="#475569" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </button>

              {wsDropdownOpen && (
                <div style={{
                  position: "absolute", top: "calc(100% + 4px)", left: 0, right: 0, zIndex: 100,
                  background: "#0f1623",
                  border: "1px solid rgba(255,255,255,0.1)",
                  borderRadius: 8,
                  overflow: "hidden",
                  boxShadow: "0 8px 32px rgba(0,0,0,0.5)",
                }}>
                  {workspaces.length === 0 && (
                    <div style={{ padding: "10px 14px", fontSize: 12, color: "#334155" }}>
                      No workspaces yet
                    </div>
                  )}
                  {workspaces.map(ws => (
                    <button
                      key={ws}
                      className="action-btn ws-option"
                      onClick={() => { setSelectedWorkspace(ws); setWsDropdownOpen(false); }}
                      style={{
                        width: "100%", padding: "10px 14px", textAlign: "left",
                        fontSize: 13,
                        color: ws === selectedWorkspace ? "#818cf8" : "#94a3b8",
                        background: ws === selectedWorkspace ? "rgba(99,102,241,0.12)" : "transparent",
                        borderBottom: "1px solid rgba(255,255,255,0.04)",
                        display: "flex", alignItems: "center", gap: 8,
                      }}
                    >
                      {ws === selectedWorkspace && (
                        <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
                          <path d="M2 5l2.5 2.5L8 3" stroke="#818cf8" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                        </svg>
                      )}
                      <span style={{ marginLeft: ws === selectedWorkspace ? 0 : 18 }}>{ws}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>

            {/* Create workspace */}
            <div style={{ display: "flex", gap: 6, marginTop: 10 }}>
              <input
                value={newWorkspace}
                onChange={e => setNewWorkspace(e.target.value)}
                onKeyDown={e => e.key === "Enter" && handleCreateWorkspace()}
                placeholder="new workspace…"
                style={{
                  flex: 1, padding: "8px 10px", fontSize: 12,
                  background: "rgba(255,255,255,0.03)",
                  border: "1px solid rgba(255,255,255,0.07)",
                  borderRadius: 7, color: "#e2e8f0",
                  transition: "border-color 0.2s",
                }}
              />
              <button
                className="action-btn"
                onClick={handleCreateWorkspace}
                style={{
                  padding: "8px 12px", borderRadius: 7,
                  background: "rgba(99,102,241,0.2)",
                  color: "#818cf8", fontSize: 13,
                }}
              >+</button>
            </div>

            {/* Delete */}
            {selectedWorkspace && (
              <button
                className="action-btn"
                onClick={handleDeleteWorkspace}
                style={{
                  width: "100%", marginTop: 8, padding: "7px",
                  borderRadius: 7, fontSize: 11,
                  background: "rgba(239,68,68,0.08)",
                  color: "#f87171",
                  letterSpacing: "0.06em",
                }}
              >
                Delete "{selectedWorkspace}"
              </button>
            )}

            {error && (
              <div style={{
                marginTop: 10, padding: "8px 10px", borderRadius: 7, fontSize: 11,
                background: "rgba(239,68,68,0.1)", color: "#f87171",
                border: "1px solid rgba(239,68,68,0.2)",
              }}>
                {error}
                <span
                  style={{ float: "right", cursor: "pointer", opacity: 0.6 }}
                  onClick={() => setError(null)}
                >✕</span>
              </div>
            )}
          </div>

          {/* Upload */}
          {selectedWorkspace && (
            <div style={{ padding: "16px 20px 0" }}>
              <div style={{ fontSize: 10, color: "#475569", letterSpacing: "0.12em", textTransform: "uppercase", marginBottom: 8 }}>
                Upload Documents
              </div>
              <label style={{
                display: "block", padding: "12px", borderRadius: 8, textAlign: "center",
                border: "1px dashed rgba(255,255,255,0.12)", cursor: "pointer", fontSize: 11,
                color: files ? "#818cf8" : "#475569",
                transition: "all 0.2s",
              }}>
                {files ? `${files.length} file${files.length > 1 ? "s" : ""} selected` : "Click to select files"}
                <input type="file" multiple onChange={e => setFiles(e.target.files)} style={{ display: "none" }} />
              </label>
              {files && (
                <button
                  className="action-btn"
                  onClick={handleUpload}
                  disabled={loading}
                  style={{
                    width: "100%", marginTop: 8, padding: "9px",
                    borderRadius: 8, fontSize: 12, fontWeight: 600,
                    background: "linear-gradient(135deg, #6366f1, #8b5cf6)",
                    color: "#fff",
                  }}
                >
                  {loading ? "Uploading…" : "Upload"}
                </button>
              )}
            </div>
          )}

          {/* Document list */}
          <div style={{ flex: 1, overflowY: "auto", padding: "16px 20px" }}>
            {documents.length > 0 && (
              <div style={{ fontSize: 10, color: "#475569", letterSpacing: "0.12em", textTransform: "uppercase", marginBottom: 10 }}>
                Documents ({documents.length})
              </div>
            )}
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              {documents.map(doc => (
                <div
                  key={doc.id}
                  className="sidebar-doc"
                  style={{
                    padding: "7px 10px", borderRadius: 7, fontSize: 11,
                    color: "#64748b", background: "rgba(255,255,255,0.02)",
                    border: "1px solid rgba(255,255,255,0.05)",
                    transition: "background 0.15s",
                    display: "flex", alignItems: "center", gap: 6,
                    overflow: "hidden",
                  }}
                  title={doc.file_name}
                >
                  <span style={{ flexShrink: 0, fontSize: 10 }}>📄</span>
                  <span style={{
                    flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                  }}>
                    {doc.file_name}
                  </span>
                  <button
                    className="action-btn doc-delete-btn"
                    onClick={() => handleDeleteDocument(doc.id, doc.file_name)}
                    title="Delete document"
                    style={{
                      flexShrink: 0, width: 18, height: 18,
                      borderRadius: 4, fontSize: 10,
                      background: "rgba(239,68,68,0.15)",
                      color: "#f87171",
                      display: "flex", alignItems: "center", justifyContent: "center",
                      opacity: 0,
                      transition: "opacity 0.15s",
                    }}
                  >✕</button>
                </div>
              ))}
            </div>
          </div>
        </aside>

        {/* ── MAIN ────────────────────────────────────────────────────────── */}
        <main style={{ flex: 1, overflowY: "auto", display: "flex", flexDirection: "column" }}>

          {/* Header */}
          <div style={{
            padding: "28px 48px 24px",
            borderBottom: "1px solid rgba(255,255,255,0.06)",
            background: "rgba(255,255,255,0.01)",
            display: "flex",
            flexDirection: "column",
            gap: 14,
            minHeight: 96,
            justifyContent: "center",
          }}>
            {/* Top row: workspace title */}
            <div>
              <h1 style={{ fontSize: 28, fontWeight: 800, letterSpacing: "0.01em", lineHeight: 1.15, margin: 0 }}>
                {selectedWorkspace
                  ? <><span style={{ color: "#6366f1", marginRight: 6 }}>/</span>{selectedWorkspace}</>
                  : <span style={{ color: "#334155", fontWeight: 600, fontSize: 20 }}>Select a workspace</span>}
              </h1>
              {selectedWorkspace && (
                <div style={{ fontSize: 12, color: "#475569", marginTop: 6, letterSpacing: "0.02em" }}>
                  {documents.length} document{documents.length !== 1 ? "s" : ""} indexed
                </div>
              )}
            </div>

            {/* Tab row — only shown when there are results */}
            {masisResult && (
              <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                {["answer", "trace", "metrics"].map(tab => (
                  <button
                    key={tab}
                    className="action-btn tab-btn"
                    onClick={() => setActiveTab(tab as any)}
                    style={{
                      padding: "6px 16px", borderRadius: 6, fontSize: 11,
                      fontWeight: 600, letterSpacing: "0.08em", textTransform: "uppercase",
                      background: activeTab === tab ? "rgba(99,102,241,0.2)" : "transparent",
                      color: activeTab === tab ? "#818cf8" : "#475569",
                      border: `1px solid ${activeTab === tab ? "rgba(99,102,241,0.3)" : "rgba(255,255,255,0.06)"}`,
                      whiteSpace: "nowrap",
                    }}
                  >
                    {tab}
                    {tab === "trace" && masisResult.trace && (
                      <span style={{
                        marginLeft: 6, fontSize: 9, padding: "1px 5px", borderRadius: 10,
                        background: "rgba(99,102,241,0.3)", color: "#a5b4fc",
                      }}>
                        {masisResult.trace.length}
                      </span>
                    )}
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Content */}
          <div style={{ flex: 1, padding: "36px 40px", overflowY: "auto" }}>
          <div style={{ maxWidth: 860, width: "100%" }}>

            {!selectedWorkspace ? (
              <div style={{
                height: "60vh", display: "flex", alignItems: "center", justifyContent: "center",
                flexDirection: "column", gap: 12,
              }}>
                <div style={{ fontSize: 48, opacity: 0.1 }}>◈</div>
                <div style={{ color: "#334155", fontSize: 13 }}>Select or create a workspace to begin</div>
              </div>
            ) : (
              <>
                {/* Query Box */}
                <div style={{
                  background: "rgba(255,255,255,0.02)",
                  border: "1px solid rgba(255,255,255,0.07)",
                  borderRadius: 12, padding: 24, marginBottom: 32,
                }}>
                  <div style={{ fontSize: 10, color: "#475569", letterSpacing: "0.12em", textTransform: "uppercase", marginBottom: 12 }}>
                    Intelligence Query
                  </div>
                  <textarea
                    value={query}
                    onChange={e => setQuery(e.target.value)}
                    onKeyDown={e => e.key === "Enter" && e.metaKey && handleMasisQuery()}
                    placeholder="Ask a strategic question about your documents…"
                    rows={4}
                    style={{
                      width: "100%", padding: "14px 16px",
                      background: "rgba(255,255,255,0.03)",
                      border: "1px solid rgba(255,255,255,0.07)",
                      borderRadius: 9, color: "#e2e8f0", fontSize: 14,
                      resize: "vertical", lineHeight: 1.6,
                      transition: "border-color 0.2s",
                    }}
                  />

                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginTop: 14 }}>
                    <span style={{ fontSize: 10, color: "#334155" }}>⌘ + Enter to run</span>
                    <button
                      className="action-btn"
                      onClick={handleMasisQuery}
                      disabled={masisLoading || !query.trim()}
                      style={{
                        padding: "11px 28px", borderRadius: 9,
                        background: masisLoading
                          ? "rgba(99,102,241,0.3)"
                          : "linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)",
                        color: "#fff", fontSize: 13, fontWeight: 700,
                        letterSpacing: "0.06em",
                        display: "flex", alignItems: "center", gap: 8,
                      }}
                    >
                      {masisLoading ? (
                        <>
                          <div style={{
                            width: 12, height: 12, borderRadius: "50%",
                            border: "2px solid rgba(255,255,255,0.3)",
                            borderTopColor: "#fff",
                            animation: "spin 0.7s linear infinite",
                          }} />
                          Analyzing…
                        </>
                      ) : "Run Analysis"}
                    </button>
                  </div>

                  {masisError && (
                    <div style={{
                      marginTop: 12, padding: "10px 14px", borderRadius: 8, fontSize: 12,
                      background: "rgba(239,68,68,0.1)", color: "#f87171",
                      border: "1px solid rgba(239,68,68,0.2)",
                    }}>
                      {masisError}
                    </div>
                  )}
                </div>

                {/* Results */}
                {masisResult && (
                  <div ref={resultRef} style={{ animation: "fadeSlideIn 0.4s ease" }}>

                    {/* HITL Banner — compact, detail shown inside answer box */}
                    {masisResult.requires_human_review && (
                      <div style={{
                        padding: "10px 16px", borderRadius: 8, marginBottom: 20,
                        background: "rgba(245,158,11,0.08)",
                        border: "1px solid rgba(245,158,11,0.2)",
                        display: "flex", gap: 10, alignItems: "center",
                      }}>
                        <span style={{ fontSize: 14 }}>⚠</span>
                        <span style={{ fontSize: 12, color: "#f59e0b", fontWeight: 600 }}>
                          Human review required — best draft shown below with details
                        </span>
                      </div>
                    )}

                    {/* Stat pills */}
                    <div style={{ display: "flex", gap: 10, marginBottom: 28, flexWrap: "wrap" }}>
                      <StatPill
                        label="Status"
                        value={masisResult.status === "success" ? "SUCCESS" : "REVIEW"}
                        color={masisResult.status === "success" ? "#22c55e" : "#f59e0b"}
                      />
                      {retryCount > 0 && (
                        <StatPill label="Retries" value={String(retryCount)} color="#6366f1" />
                      )}
                      {masisResult.metrics?.citation_count !== undefined && (
                        <StatPill label="Citations" value={String(masisResult.metrics.citation_count)} color="#0ea5e9" />
                      )}
                      {masisResult.metrics?.avg_retrieval_score !== undefined && (
                        <StatPill
                          label="Avg Retrieval Score"
                          value={masisResult.metrics.avg_retrieval_score.toFixed(3)}
                          color="#8b5cf6"
                        />
                      )}
                    </div>

                    {/* ── ANSWER TAB ── */}
                    {activeTab === "answer" && (
                      <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>

                        {/* Confidence ring + answer */}
                        <div style={{
                          background: "rgba(255,255,255,0.02)",
                          border: `1px solid ${masisResult.requires_human_review ? "rgba(245,158,11,0.25)" : "rgba(255,255,255,0.07)"}`,
                          borderRadius: 12, overflow: "hidden",
                        }}>
                          {/* Header row: label + confidence ring */}
                          <div style={{
                            padding: "16px 24px",
                            borderBottom: "1px solid rgba(255,255,255,0.05)",
                            display: "flex", alignItems: "center", justifyContent: "space-between",
                          }}>
                            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                              <span style={{ fontSize: 10, color: "#475569", letterSpacing: "0.12em", textTransform: "uppercase" }}>
                                {masisResult.requires_human_review ? "Best Draft — Low Confidence" : "Final Answer"}
                              </span>
                              {masisResult.requires_human_review && (
                                <span style={{
                                  fontSize: 9, padding: "2px 8px", borderRadius: 4,
                                  background: "rgba(245,158,11,0.15)",
                                  color: "#f59e0b",
                                  letterSpacing: "0.08em", textTransform: "uppercase", fontWeight: 700,
                                }}>
                                  Review Before Using
                                </span>
                              )}
                            </div>
                            {masisResult.confidence !== undefined && (
                              <ConfidenceRing value={masisResult.confidence} />
                            )}
                          </div>

                          {/* Low-confidence inline warning strip */}
                          {masisResult.requires_human_review && (
                            <div style={{
                              padding: "10px 24px",
                              background: "rgba(245,158,11,0.06)",
                              borderBottom: "1px solid rgba(245,158,11,0.12)",
                              display: "flex", alignItems: "flex-start", gap: 10,
                            }}>
                              <span style={{ fontSize: 14, lineHeight: 1, flexShrink: 0, marginTop: 1 }}>⚠</span>
                              <span style={{ fontSize: 12, color: "#92400e", lineHeight: 1.6 }}>
                                {masisResult.clarification_question || "The system could not reach sufficient confidence. The answer below is the best draft produced — verify against the source documents before relying on it."}
                              </span>
                            </div>
                          )}

                          {/* The answer itself — always shown */}
                          <div style={{
                            padding: "24px", fontSize: 14, lineHeight: 1.8,
                            color: masisResult.requires_human_review ? "#94a3b8" : "#cbd5e1",
                            whiteSpace: "pre-wrap",
                            fontFamily: "'Syne', sans-serif",
                            opacity: masisResult.requires_human_review ? 0.85 : 1,
                          }}>
                            {masisResult.answer
                              ? masisResult.answer
                              : (
                                <div style={{
                                  display: "flex", flexDirection: "column", alignItems: "center",
                                  gap: 10, padding: "24px 0", textAlign: "center",
                                }}>
                                  <div style={{ fontSize: 32, opacity: 0.2 }}>◈</div>
                                  <span style={{ color: "#475569", fontSize: 13 }}>
                                    No documents contained enough information to generate an answer.
                                  </span>
                                  <span style={{ color: "#334155", fontSize: 11 }}>
                                    Try uploading relevant documents or refining your query.
                                  </span>
                                </div>
                              )
                            }
                          </div>
                        </div>

                        {/* Quality Assessment — always shown when available, including on low-confidence HITL responses */}
                        {masisResult.evaluation && (
                          <div style={{
                            background: masisResult.requires_human_review ? "rgba(245,158,11,0.04)" : "rgba(255,255,255,0.02)",
                            border: `1px solid ${masisResult.requires_human_review ? "rgba(245,158,11,0.2)" : "rgba(255,255,255,0.07)"}`,
                            borderRadius: 12, padding: "20px 24px",
                          }}>
                            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 18 }}>
                              <span style={{ fontSize: 10, color: "#475569", letterSpacing: "0.12em", textTransform: "uppercase" }}>
                                Quality Assessment
                              </span>
                              {masisResult.requires_human_review && (
                                <span style={{ fontSize: 10, color: "#f59e0b" }}>
                                  — explains why confidence was insufficient
                                </span>
                              )}
                            </div>
                            <ScoreBar label="Faithfulness" value={masisResult.evaluation.faithfulness} delay={0} />
                            <ScoreBar label="Relevance" value={masisResult.evaluation.relevance} delay={100} />
                            <ScoreBar label="Completeness" value={masisResult.evaluation.completeness} delay={200} />
                            <ScoreBar label="Reasoning Quality" value={masisResult.evaluation.reasoning_quality} delay={300} />

                            {masisResult.evaluation.improvement_suggestions?.length > 0 && (
                              <div style={{ marginTop: 18, paddingTop: 16, borderTop: "1px solid rgba(255,255,255,0.05)" }}>
                                <div style={{ fontSize: 10, color: "#475569", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 10 }}>
                                  Improvement Suggestions
                                </div>
                                <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                                  {masisResult.evaluation.improvement_suggestions.map((s, i) => (
                                    <div key={i} style={{ fontSize: 12, color: "#64748b", display: "flex", gap: 8 }}>
                                      <span style={{ color: "#334155", flexShrink: 0 }}>→</span>
                                      {s}
                                    </div>
                                  ))}
                                </div>
                              </div>
                            )}
                          </div>
                        )}

                        {/* Critique (if issues found) */}
                        {masisResult.critique && (
                          masisResult.critique.hallucination_detected ||
                          masisResult.critique.unsupported_claims?.length > 0 ||
                          masisResult.critique.conflicting_evidence?.length > 0
                        ) && (
                          <div style={{
                            background: "rgba(239,68,68,0.04)",
                            border: "1px solid rgba(239,68,68,0.15)",
                            borderRadius: 12, padding: "20px 24px",
                          }}>
                            <div style={{ fontSize: 10, color: "#f87171", letterSpacing: "0.12em", textTransform: "uppercase", marginBottom: 14 }}>
                              ⚑ Critic Findings
                            </div>
                            {masisResult.critique.hallucination_detected && (
                              <div style={{ fontSize: 12, color: "#f87171", marginBottom: 10 }}>
                                ✗ Hallucination detected in final answer
                              </div>
                            )}
                            {masisResult.critique.unsupported_claims?.map((c, i) => (
                              <div key={i} style={{ fontSize: 12, color: "#94a3b8", marginBottom: 4 }}>
                                <span style={{ color: "#f59e0b" }}>Unsupported: </span>{c}
                              </div>
                            ))}
                            {masisResult.critique.conflicting_evidence?.map((c, i) => (
                              <div key={i} style={{ fontSize: 12, color: "#94a3b8", marginBottom: 4 }}>
                                <span style={{ color: "#ef4444" }}>Conflict: </span>{c}
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    )}

                    {/* ── TRACE TAB ── */}
                    {activeTab === "trace" && (
                      <div style={{
                        background: "rgba(255,255,255,0.02)",
                        border: "1px solid rgba(255,255,255,0.07)",
                        borderRadius: 12, padding: "24px",
                      }}>
                        <div style={{ fontSize: 10, color: "#475569", letterSpacing: "0.12em", textTransform: "uppercase", marginBottom: 20 }}>
                          Execution Trace — {masisResult.trace?.length ?? 0} events
                        </div>
                        <div>
                          {masisResult.trace?.map((entry, i) => (
                            <TraceEntry key={i} entry={entry} index={i} />
                          ))}
                        </div>
                      </div>
                    )}

                    {/* ── METRICS TAB ── */}
                    {activeTab === "metrics" && masisResult.metrics && (
                      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>

                        {/* Node latencies */}
                        {masisResult.metrics.node_latency_ms && Object.keys(masisResult.metrics.node_latency_ms).length > 0 && (
                          <MetricCard title="Node Latencies">
                            {Object.entries(masisResult.metrics.node_latency_ms).map(([node, ms]) => (
                              <div key={node} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
                                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                                  <div style={{
                                    width: 8, height: 8, borderRadius: "50%",
                                    background: NODE_COLORS[node] || "#64748b",
                                  }} />
                                  <span style={{ fontSize: 12, color: "#64748b", textTransform: "capitalize" }}>{node}</span>
                                </div>
                                <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 13, color: "#94a3b8" }}>
                                  {ms} ms
                                </span>
                              </div>
                            ))}
                          </MetricCard>
                        )}

                        {/* Confidence history */}
                        {masisResult.metrics.confidence_history?.length > 0 && (
                          <MetricCard title="Confidence History">
                            <div style={{ display: "flex", gap: 10, alignItems: "flex-end", height: 60 }}>
                              {masisResult.metrics.confidence_history.map((v, i) => {
                                const meta = confidenceMeta(v)
                                return (
                                  <div key={i} style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 4, flex: 1 }}>
                                    <span style={{ fontSize: 10, fontFamily: "'JetBrains Mono', monospace", color: meta.color }}>
                                      {(v * 100).toFixed(0)}%
                                    </span>
                                    <div style={{
                                      width: "100%", height: `${v * 50}px`,
                                      background: meta.color, borderRadius: 3, opacity: 0.7,
                                      minHeight: 4,
                                    }} />
                                    <span style={{ fontSize: 9, color: "#334155" }}>iter {i + 1}</span>
                                  </div>
                                )
                              })}
                            </div>
                          </MetricCard>
                        )}

                        {/* Citation violations */}
                        {masisResult.metrics.citation_violations?.length > 0 && (
                          <MetricCard title="Citation Engine Report">
                            {masisResult.metrics.citation_violations.map((v: any, i: number) => (
                              <div key={i} style={{ marginBottom: i < masisResult.metrics.citation_violations.length - 1 ? 12 : 0 }}>
                                <div style={{ fontSize: 10, color: "#475569", marginBottom: 6 }}>Iteration {v.iteration + 1}</div>
                                <div style={{ display: "flex", gap: 10 }}>
                                  <Chip label="invalid IDs" value={v.invalid_ids?.length ?? 0} danger={v.invalid_ids?.length > 0} />
                                  <Chip label="uncited claims" value={v.uncited_claims ?? 0} danger={v.uncited_claims > 0} />
                                </div>
                              </div>
                            ))}
                          </MetricCard>
                        )}

                        {/* Raw metrics dump */}
                        <details style={{
                          background: "rgba(255,255,255,0.02)",
                          border: "1px solid rgba(255,255,255,0.07)",
                          borderRadius: 12, overflow: "hidden",
                        }}>
                          <summary style={{
                            padding: "14px 20px", cursor: "pointer", fontSize: 11,
                            color: "#475569", letterSpacing: "0.1em", textTransform: "uppercase",
                            userSelect: "none",
                          }}>
                            Raw Metrics JSON
                          </summary>
                          <pre style={{
                            padding: "16px 20px", fontSize: 11,
                            fontFamily: "'JetBrains Mono', monospace",
                            color: "#475569", overflowX: "auto",
                            borderTop: "1px solid rgba(255,255,255,0.05)",
                            lineHeight: 1.6,
                          }}>
                            {JSON.stringify(masisResult.metrics, null, 2)}
                          </pre>
                        </details>
                      </div>
                    )}
                  </div>
                )}
              </>
            )}
          </div>
          </div>
        </main>
      </div>
    </>
  )
}

// ─── Tiny shared components ───────────────────────────────────────────────────

function StatPill({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div style={{
      display: "inline-flex", alignItems: "center", gap: 8,
      padding: "6px 12px", borderRadius: 20,
      background: `${color}12`,
      border: `1px solid ${color}30`,
    }}>
      <span style={{ fontSize: 10, color: "#475569", letterSpacing: "0.08em", textTransform: "uppercase" }}>{label}</span>
      <span style={{ fontSize: 11, fontFamily: "'JetBrains Mono', monospace", color, fontWeight: 700 }}>{value}</span>
    </div>
  )
}

function MetricCard({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={{
      background: "rgba(255,255,255,0.02)",
      border: "1px solid rgba(255,255,255,0.07)",
      borderRadius: 12, padding: "20px 24px",
    }}>
      <div style={{ fontSize: 10, color: "#475569", letterSpacing: "0.12em", textTransform: "uppercase", marginBottom: 16 }}>
        {title}
      </div>
      {children}
    </div>
  )
}

export default App