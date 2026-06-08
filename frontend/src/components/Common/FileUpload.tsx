import { useState, useRef, DragEvent } from 'react';

interface FileUploadProps {
  accept?: string;
  multiple?: boolean;
  maxSize?: number;
  onFilesSelected: (files: File[]) => void;
  label?: string;
}

export function FileUpload({
  accept = '.json,.csv,.txt',
  multiple = false,
  maxSize = 100 * 1024 * 1024,
  onFilesSelected,
  label = '上传文件',
}: FileUploadProps) {
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const files = Array.from(e.dataTransfer.files).filter(
      (f) => f.size <= maxSize,
    );
    if (files.length > 0) onFilesSelected(files);
  };

  const handleClick = () => {
    inputRef.current?.click();
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files ? Array.from(e.target.files) : [];
    if (files.length > 0) onFilesSelected(files);
    if (inputRef.current) inputRef.current.value = '';
  };

  return (
    <div
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={handleClick}
      className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-colors ${
        isDragging
          ? 'border-primary-500 bg-primary-500/10'
          : 'border-surface-700 hover:border-surface-600 bg-surface-800/50'
      }`}
    >
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        multiple={multiple}
        onChange={handleChange}
        className="hidden"
      />
      <div className="text-2xl mb-2">{isDragging ? '📂' : '📁'}</div>
      <p className="text-sm text-gray-300">{isDragging ? '松开以上传' : label}</p>
      <p className="text-xs text-gray-500 mt-1">
        支持 {accept} · 最大 {Math.round(maxSize / 1024 / 1024)}MB
      </p>
    </div>
  );
}
