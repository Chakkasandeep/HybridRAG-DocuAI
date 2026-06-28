import React, { useState, useRef } from 'react';
import { Upload, RefreshCw, AlertCircle, FileText, CheckCircle2 } from 'lucide-react';
import api from '../utils/api';

const UploadSection = ({ onUploadSuccess, onProcessSuccess }) => {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [isUploading, setIsUploading] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [statusMsg, setStatusMsg] = useState({ type: '', text: '' });
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    if (e.target.files) {
      setSelectedFiles(Array.from(e.target.files));
      setStatusMsg({ type: '', text: '' });
    }
  };

  const handleUploadAndProcess = async () => {
    if (selectedFiles.length === 0) {
      setStatusMsg({ type: 'error', text: 'Please select at least one PDF file.' });
      return;
    }

    const formData = new FormData();
    selectedFiles.forEach((file) => {
      formData.append('files', file);
    });

    setIsUploading(true);
    setStatusMsg({ type: 'info', text: 'Uploading files...' });

    try {
      // 1. Upload files
      const uploadRes = await api.post('/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setIsUploading(false);
      onUploadSuccess(uploadRes.data);

      // 2. Process / Index files
      setIsProcessing(true);
      setStatusMsg({ type: 'info', text: 'Indexing documents (extracting text & generating FAISS embeddings)...' });
      
      const processRes = await api.post('/process');
      setIsProcessing(false);
      
      setStatusMsg({
        type: 'success',
        text: `Indexed successfully! Generated ${processRes.data.num_chunks} chunks.`
      });
      setSelectedFiles([]);
      if (fileInputRef.current) fileInputRef.current.value = '';
      onProcessSuccess();
    } catch (err) {
      setIsUploading(false);
      setIsProcessing(false);
      const errMsg = err.response?.data?.detail || 'An error occurred during processing.';
      setStatusMsg({ type: 'error', text: errMsg });
    }
  };

  return (
    <div className="space-y-4 p-4 border border-chat-border rounded-lg bg-chat-bg">
      <div className="flex flex-col items-center justify-center border-2 border-dashed border-chat-border hover:border-blue-500 rounded-lg p-6 cursor-pointer bg-chat-sidebar transition-colors relative">
        <input
          type="file"
          multiple
          accept=".pdf"
          onChange={handleFileChange}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          ref={fileInputRef}
          disabled={isUploading || isProcessing}
        />
        <Upload className="w-8 h-8 text-zinc-400 mb-2" />
        <span className="text-xs text-zinc-300 font-medium">Click to upload PDFs</span>
        <span className="text-[10px] text-zinc-500 mt-1">Accepts multiple files</span>
      </div>

      {selectedFiles.length > 0 && (
        <div className="max-h-24 overflow-y-auto space-y-1">
          {selectedFiles.map((file, idx) => (
            <div key={idx} className="flex items-center text-xs text-zinc-300 bg-chat-card p-1.5 rounded border border-chat-border">
              <FileText className="w-3.5 h-3.5 mr-1.5 text-zinc-400 flex-shrink-0" />
              <span className="truncate flex-1">{file.name}</span>
              <span className="text-[10px] text-zinc-500 ml-1">{(file.size / 1024).toFixed(0)} KB</span>
            </div>
          ))}
        </div>
      )}

      {selectedFiles.length > 0 && (
        <button
          onClick={handleUploadAndProcess}
          disabled={isUploading || isProcessing}
          className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-blue-800 disabled:opacity-50 text-white py-2 px-3 rounded-lg text-xs font-semibold flex items-center justify-center transition-colors"
        >
          {(isUploading || isProcessing) ? (
            <RefreshCw className="w-3.5 h-3.5 mr-1.5 animate-spin" />
          ) : null}
          {isUploading ? 'Uploading...' : isProcessing ? 'Processing...' : 'Submit & Process'}
        </button>
      )}

      {statusMsg.text && (
        <div className={`p-2.5 rounded-lg text-xs flex items-start border ${
          statusMsg.type === 'error' ? 'bg-red-950/30 text-red-400 border-red-900/50' :
          statusMsg.type === 'success' ? 'bg-emerald-950/30 text-emerald-400 border-emerald-900/50' :
          'bg-blue-950/30 text-blue-400 border-blue-900/50'
        }`}>
          {statusMsg.type === 'error' && <AlertCircle className="w-3.5 h-3.5 mr-1.5 mt-0.5 flex-shrink-0" />}
          {statusMsg.type === 'success' && <CheckCircle2 className="w-3.5 h-3.5 mr-1.5 mt-0.5 flex-shrink-0" />}
          {statusMsg.type === 'info' && <RefreshCw className="w-3.5 h-3.5 mr-1.5 mt-0.5 animate-spin flex-shrink-0" />}
          <span className="flex-1 leading-relaxed">{statusMsg.text}</span>
        </div>
      )}
    </div>
  );
};

export default UploadSection;
