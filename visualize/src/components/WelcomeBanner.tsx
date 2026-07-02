import React, { useRef, useState } from "react"
import { FolderOpen, Sparkles, UploadCloud } from "lucide-react"

interface WelcomeBannerProps {
  onFilesSelected: (files: File[]) => void
  onLoadDemo: () => void
  isLoading: boolean
}

export function WelcomeBanner({
  onFilesSelected,
  onLoadDemo,
  isLoading,
}: WelcomeBannerProps) {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [isDragOver, setIsDragOver] = useState(false)

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(true)
  }

  const handleDragLeave = () => {
    setIsDragOver(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      // Convert FileList to array
      const filesArr = Array.from(e.dataTransfer.files)
      onFilesSelected(filesArr)
    }
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      onFilesSelected(Array.from(e.target.files))
    }
  }

  const triggerPicker = () => {
    fileInputRef.current?.click()
  }

  return (
    <div
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      className={`rounded-2xl border border-dashed p-12 text-center flex flex-col items-center justify-center gap-6 backdrop-blur-md transition-all duration-300 ${
        isDragOver
          ? "border-blue-500 bg-blue-500/5 shadow-[0_0_30px_rgba(59,130,246,0.15)]"
          : "border-white/10 bg-white/[0.02]"
      }`}
    >
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        style={{ display: "none" }}
        {...({
          webkitdirectory: "",
          directory: "",
        } as any)}
        multiple
      />

      <div className="w-20 h-20 bg-blue-500/10 text-blue-500 rounded-full flex items-center justify-center text-3xl animate-pulse shadow-[0_0_20px_rgba(59,130,246,0.1)]">
        <UploadCloud className="h-10 w-10" />
      </div>

      <div className="flex flex-col gap-2 max-w-lg">
        <h2 className="font-heading font-semibold text-2xl text-white">
          Load Pipeline Workspace
        </h2>
        <p className="text-gray-400 text-sm leading-relaxed">
          Select your project's workspace folder (or drag and drop it here). We will
          recursively load SFT prompts, raw/compressed traces, training splits, and evaluation
          results to let you inspect them.
        </p>
      </div>

      <div className="flex gap-4 mt-2">
        <button
          onClick={triggerPicker}
          disabled={isLoading}
          className="flex items-center gap-2 px-5 py-2.5 rounded-lg bg-gradient-to-r from-blue-500 to-indigo-600 hover:from-blue-600 hover:to-indigo-700 text-white font-medium text-sm transition-all duration-200 cursor-pointer shadow-lg shadow-blue-500/20 disabled:opacity-50"
        >
          <FolderOpen className="h-4 w-4" />
          {isLoading ? "Reading Folder..." : "Select Workspace Folder"}
        </button>

        <button
          onClick={onLoadDemo}
          disabled={isLoading}
          className="flex items-center gap-2 px-5 py-2.5 rounded-lg border border-white/10 hover:border-white/20 bg-white/5 hover:bg-white/10 text-white font-medium text-sm transition-all duration-200 cursor-pointer disabled:opacity-50"
        >
          <Sparkles className="h-4 w-4 text-amber-400" />
          Load Demo Data
        </button>
      </div>
    </div>
  )
}
