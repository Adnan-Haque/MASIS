import { useEffect, useState } from "react"
import { api } from "./api/temp"
import type { Document } from "./types"
import axios from "axios"

function App() {
  const [workspaces, setWorkspaces] = useState<string[]>([])
  const [selectedWorkspace, setSelectedWorkspace] =
    useState<string>("")
  const [documents, setDocuments] = useState<Document[]>([])
  const [files, setFiles] = useState<FileList | null>(null)
  const [newWorkspace, setNewWorkspace] = useState("")
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

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

  // ================= FETCH DOCUMENTS =================

  const fetchDocuments = async (workspace: string) => {
    try {
      const res = await api.get(
        `/workspaces/${workspace}/documents`
      )
      setDocuments(res.data)
    } catch {
      setError("Failed to load documents")
    }
  }

  // ================= CREATE WORKSPACE =================

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

  // ================= DELETE WORKSPACE =================

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

  // ================= DELETE DOCUMENT =================

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

  // ================= UPLOAD =================

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

  // ================= SMART POLLING =================

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

  // ================= STATUS STYLE =================

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

        {/* Error */}
        {error && (
          <div className="bg-red-100 text-red-700 p-2 rounded mb-4 text-sm">
            {error}
          </div>
        )}

        {/* Workspace Dropdown */}
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

        {/* Create Workspace */}
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

        {/* GLOBAL PROCESSING INDICATOR */}
        {anyProcessing && (
          <div className="mb-4 text-sm text-yellow-600 font-medium">
            Processing documents...
          </div>
        )}

        {/* DOCUMENT LIST */}
        <div className="flex-1 overflow-y-auto space-y-3">
          {documents.map((doc) => {
            const percent =
              doc.total_chunks > 0
                ? Math.floor(
                    (doc.processed_chunks /
                      doc.total_chunks) *
                      100
                  )
                : 0

            return (
              <div
                key={doc.id}
                className="p-3 border rounded bg-gray-50"
              >
                <div className="flex justify-between items-center">
                  <span className="truncate text-sm font-medium">
                    {doc.file_name}
                  </span>

                  <div className="flex items-center gap-2">
                    <span
                      className={`text-xs px-2 py-1 rounded ${statusStyle(
                        doc.status
                      )}`}
                    >
                      {doc.status}
                    </span>

                    <button
                      onClick={() =>
                        handleDeleteDocument(doc.id)
                      }
                      className="text-gray-400 hover:text-red-500"
                    >
                      ✕
                    </button>
                  </div>
                </div>

                {doc.status === "PROCESSING" &&
                  doc.total_chunks > 0 && (
                    <div className="mt-2">
                      <div className="w-full bg-gray-200 h-2 rounded">
                        <div
                          className="bg-blue-600 h-2 rounded transition-all duration-300"
                          style={{
                            width: `${percent}%`
                          }}
                        />
                      </div>
                      <div className="text-xs text-gray-500 mt-1">
                        {percent}% • {doc.processed_chunks} /{" "}
                        {doc.total_chunks}
                      </div>
                    </div>
                  )}
              </div>
            )
          })}
        </div>
      </div>

      {/* MAIN AREA */}
      <div className="flex-1 p-10 bg-gray-50">
        {selectedWorkspace ? (
          <>
            <h2 className="text-3xl font-bold mb-6">
              {selectedWorkspace}
            </h2>

            <div className="bg-white p-6 rounded shadow">
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
                className="bg-blue-600 text-white px-4 py-2 rounded disabled:opacity-50"
              >
                {loading
                  ? "Uploading..."
                  : "Upload"}
              </button>
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