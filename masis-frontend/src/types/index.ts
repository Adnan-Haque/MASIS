export interface Document {
  id: string
  file_name: string
  status: "READY" | "PROCESSING" | "FAILED"
}