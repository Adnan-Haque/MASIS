import { useEffect, useState } from "react"
import { api } from "./api/temp"
import type { Document } from "./types"
import axios from "axios"

function App() {
  const [workspaces, setWorkspaces] = useState<string[]>([])
  const [selectedWorkspace, setSelectedWorkspace] = useState<string>("")
  const [documents, setDocuments] = useState<Document[]>([])
  const [files, setFiles] = useState<FileList | null>(null)
  const [newWorkspace, setNewWorkspace] = useState("")
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // 🔥 MASIS STATE
  const [query, setQuery] = useState("")
  const [masisResult, setMasisResult] = useState<any>(null)
  const [masisLoading, setMasisLoading] = useState(false)
  const [masisError, setMasisError] = useState<string | null>(null)

  // ================= FETCH WORKSPACES =================

  const fetchWorkspaces = async () => {
    try {
      const res = await api.get("/workspaces")
      setWorkspaces(res.data)
      if (!selectedWorkspace && res.data.length > 0) {
        setSelectedWorkspace(res.data[0])
      }
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

  const handleCreateWorkspace = async () => {
    if (!newWorkspace.trim()) {
      setError("Workspace name cannot be empty")
      return
    }

    try {
      await api.post(`/workspaces/${newWorkspace}`)
      setSelectedWorkspace(newWorkspace)
      setNewWorkspace("")
      setError(null)
      fetchWorkspaces()
    } catch (err) {
      if (axios.isAxiosError(err)) {
        if (err.response?.status === 409) {
          setError("Workspace already exists")
        } else {
          setError("Failed to create workspace")
        }
      }
    }
  }

  const handleDeleteWorkspace = async () => {
    try {
      await api.delete(`/workspaces/${selectedWorkspace}`)
      setSelectedWorkspace("")
      setDocuments([])
      fetchWorkspaces()
    } catch {
      setError("Failed to delete workspace")
    }
  }

  const handleDeleteDocument = async (docId: string) => {
    try {
      await api.delete(
        `/workspaces/${selectedWorkspace}/documents/${docId}`
      )
      fetchDocuments(selectedWorkspace)
    } catch {
      setError("Failed to delete document")
    }
  }

  const handleUpload = async () => {
    if (!files || !selectedWorkspace) return
    setLoading(true)
    setError(null)

    for (const file of Array.from(files)) {
      try {
        const formData = new FormData()
        formData.append("file", file)

        await api.post(
          `/workspaces/${selectedWorkspace}/upload`,
          formData
        )
      } catch (err) {
        if (axios.isAxiosError(err)) {
          if (err.response?.status === 409) {
            setError(`Duplicate file: ${file.name}`)
          } else {
            setError(`Upload failed: ${file.name}`)
          }
        }
      }
    }

    setFiles(null)
    fetchDocuments(selectedWorkspace)
    setLoading(false)
  }

  const handleMasisQuery = async () => {
    if (!selectedWorkspace || !query.trim()) {
      setMasisError("Enter a query and select a workspace")
      return
    }

    setMasisLoading(true)
    setMasisError(null)
    setMasisResult(null)

    try {
      const res = await api.post(
        `/masis/workspaces/${selectedWorkspace}`,
        { query }
      )
      setMasisResult(res.data)
    } catch {
      setMasisError("MASIS query failed")
    }

    setMasisLoading(false)
  }

  const anyProcessing = documents.some(
    (d) => d.status === "PROCESSING"
  )

  useEffect(() => {
    if (!selectedWorkspace) return
    fetchDocuments(selectedWorkspace)

    if (!anyProcessing) return
    const interval = setInterval(() => {
      fetchDocuments(selectedWorkspace)
    }, 2000)

    return () => clearInterval(interval)
  }, [selectedWorkspace, anyProcessing])

  useEffect(() => {
    fetchWorkspaces()
  }, [])

  const statusStyle = (status: string) => {
    switch (status) {
      case "READY":
        return "bg-green-100 text-green-700"
      case "PROCESSING":
        return "bg-yellow-100 text-yellow-700"
      case "FAILED":
        return "bg-red-100 text-red-700"
      default:
        return "bg-gray-100 text-gray-700"
    }
  }

  return (
    <div className="flex h-screen font-sans">
      {/* SIDEBAR */}
      <div className="w-80 bg-white border-r p-6 flex flex-col">
        <h1 className="text-2xl font-semibold mb-6">
          MASIS
        </h1>

        {error && (
          <div className="bg-red-100 text-red-700 p-2 rounded mb-4 text-sm">
            {error}
          </div>
        )}

        <select
          value={selectedWorkspace}
          onChange={(e) =>
            setSelectedWorkspace(e.target.value)
          }
          className="w-full p-2 border rounded mb-3"
        >
          <option value="">Select Workspace</option>
          {workspaces.map((ws) => (
            <option key={ws} value={ws}>
              {ws}
            </option>
          ))}
        </select>

        <input
          type="text"
          placeholder="New workspace"
          value={newWorkspace}
          onChange={(e) =>
            setNewWorkspace(e.target.value)
          }
          className="w-full p-2 border rounded mb-2"
        />

        <button
          onClick={handleCreateWorkspace}
          className="w-full bg-blue-600 text-white py-2 rounded mb-3"
        >
          Create Workspace
        </button>

        {selectedWorkspace && (
          <button
            onClick={handleDeleteWorkspace}
            className="w-full bg-red-500 text-white py-2 rounded mb-4"
          >
            Delete Workspace
          </button>
        )}

        <div className="flex-1 overflow-y-auto space-y-3">
          {documents.map((doc) => (
            <div
              key={doc.id}
              className="p-3 border rounded bg-gray-50"
            >
              <div className="flex justify-between items-center">
                <span className="truncate text-sm font-medium">
                  {doc.file_name}
                </span>

                <span
                  className={`text-xs px-2 py-1 rounded ${statusStyle(
                    doc.status
                  )}`}
                >
                  {doc.status}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* MAIN AREA */}
      <div className="flex-1 p-10 bg-gray-50">
        {selectedWorkspace ? (
          <>
            <h2 className="text-3xl font-bold mb-6">
              {selectedWorkspace}
            </h2>

            <div className="bg-white p-6 rounded shadow mb-8">
              <input
                type="file"
                multiple
                onChange={(e) =>
                  setFiles(e.target.files)
                }
                className="mb-4"
              />

              <button
                onClick={handleUpload}
                disabled={loading}
                className="bg-blue-600 text-white px-4 py-2 rounded"
              >
                {loading ? "Uploading..." : "Upload"}
              </button>
            </div>

            {/* MASIS QUERY */}
            <div className="bg-white p-6 rounded shadow">
              <h3 className="text-xl font-semibold mb-4">
                Strategic Intelligence Query
              </h3>

              <textarea
                value={query}
                onChange={(e) =>
                  setQuery(e.target.value)
                }
                placeholder="Ask a strategic question..."
                className="w-full border rounded p-3 mb-4"
                rows={4}
              />

              <button
                onClick={handleMasisQuery}
                disabled={masisLoading}
                className="bg-purple-600 text-white px-4 py-2 rounded"
              >
                {masisLoading ? "Analyzing..." : "Run MASIS"}
              </button>

              {masisResult && (
                <div className="mt-6 space-y-6">

                  {/* HITL */}
                  {masisResult.requires_human_review && (
                    <div className="bg-orange-100 border border-orange-300 text-orange-800 p-4 rounded">
                      ⚠ {masisResult.clarification_question}
                    </div>
                  )}

                  {/* FINAL ANSWER */}
                  <div>
                    <h4 className="font-semibold mb-2">
                      Final Answer
                    </h4>
                    <div className="bg-gray-100 p-4 rounded whitespace-pre-wrap">
                      {masisResult.answer}
                    </div>
                  </div>

                  {/* CONFIDENCE */}
                  <div>
                    <h4 className="font-semibold mb-2">
                      Confidence
                    </h4>
                    <span className="px-3 py-1 rounded bg-blue-100 text-blue-700">
                      {(masisResult.confidence * 100).toFixed(1)}%
                    </span>
                  </div>

                  {/* METRICS */}
                  {masisResult.metrics && (
                    <details className="bg-white border rounded p-4">
                      <summary className="cursor-pointer font-semibold">
                        System Metrics
                      </summary>

                      <div className="mt-3 text-sm space-y-2">
                        <div>
                          Avg Retrieval Score:{" "}
                          {masisResult.metrics.avg_retrieval_score?.toFixed(2)}
                        </div>

                        <div>
                          Citations Used:{" "}
                          {masisResult.metrics.citation_count}
                        </div>

                        {masisResult.metrics.node_latency_ms &&
                          Object.entries(
                            masisResult.metrics.node_latency_ms
                          ).map(([node, time]) => (
                            <div key={node}>
                              {node}: {time} ms
                            </div>
                          ))}
                      </div>
                    </details>
                  )}
                </div>
              )}
            </div>
          </>
        ) : (
          <div className="text-gray-500">
            Select a workspace
          </div>
        )}
      </div>
    </div>
  )
}

export default App