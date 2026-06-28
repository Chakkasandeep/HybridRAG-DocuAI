import React, { useState, useEffect } from 'react';
import { Layers, Plus, MessageSquare, BarChart3, Key, Eye, EyeOff, Trash2, FileText, Check, AlertCircle } from 'lucide-react';
import UploadSection from './UploadSection';
import api from '../utils/api';

const Sidebar = ({
  activePage,
  setActivePage,
  documents,
  setDocuments,
  onClearChat,
  onUploadSuccess,
  onProcessSuccess,
  fetchDocuments
}) => {
  const [apiKey, setApiKey] = useState('');
  const [showKey, setShowKey] = useState(false);
  const [isDeleting, setIsDeleting] = useState(null);
  const [keySaved, setKeySaved] = useState(false);

  useEffect(() => {
    const savedKey = localStorage.getItem('groq_api_key') || '';
    setApiKey(savedKey);
    if (savedKey) {
      setKeySaved(true);
    }
  }, []);

  const handleSaveKey = (e) => {
    const value = e.target.value;
    setApiKey(value);
    if (value.trim()) {
      localStorage.setItem('groq_api_key', value.trim());
      setKeySaved(true);
    } else {
      localStorage.removeItem('groq_api_key');
      setKeySaved(false);
    }
  };

  const handleDeleteDoc = async (filename) => {
    if (confirm(`Are you sure you want to delete ${filename}? This will remove it from the FAISS index.`)) {
      setIsDeleting(filename);
      try {
        await api.delete(`/document/${encodeURIComponent(filename)}`);
        await fetchDocuments();
      } catch (err) {
        console.error("Error deleting document:", err);
        alert(err.response?.data?.detail || "Failed to delete document.");
      } finally {
        setIsDeleting(null);
      }
    }
  };

  return (
    <div className="w-80 bg-chat-sidebar border-r border-chat-border flex flex-col h-full flex-shrink-0 select-none">
      {/* Title Logo */}
      <div className="h-14 border-b border-chat-border flex items-center px-5 space-x-3 bg-chat-sidebar/50">
        <div className="bg-blue-600 p-1.5 rounded-lg text-white">
          <Layers className="w-5 h-5" />
        </div>
        <div>
          <h1 className="text-sm font-bold text-zinc-100 tracking-tight">DocuAI Assistant</h1>
          <span className="text-[9px] text-zinc-500 font-semibold tracking-wider uppercase">Hybrid RAG System</span>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-6">
        {/* Navigation Mode */}
        <div className="space-y-1.5">
          <span className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider px-1">Navigation</span>
          <button
            onClick={() => setActivePage('chat')}
            className={`w-full flex items-center space-x-3 px-3 py-2.5 rounded-lg text-xs font-semibold transition-all ${
              activePage === 'chat'
                ? 'bg-blue-600 text-white shadow'
                : 'text-zinc-400 hover:text-zinc-200 hover:bg-chat-card/50'
            }`}
          >
            <MessageSquare className="w-4 h-4" />
            <span>Chat with PDF</span>
          </button>
          <button
            onClick={() => setActivePage('evaluation')}
            className={`w-full flex items-center space-x-3 px-3 py-2.5 rounded-lg text-xs font-semibold transition-all ${
              activePage === 'evaluation'
                ? 'bg-blue-600 text-white shadow'
                : 'text-zinc-400 hover:text-zinc-200 hover:bg-chat-card/50'
            }`}
          >
            <BarChart3 className="w-4 h-4" />
            <span>Evaluation Dashboard</span>
          </button>
        </div>

        {/* Clear Chat Button (only visible in chat page) */}
        {activePage === 'chat' && (
          <button
            onClick={onClearChat}
            className="w-full flex items-center justify-center space-x-2 px-3 py-2 border border-chat-border hover:bg-chat-card/50 text-zinc-300 hover:text-white rounded-lg text-xs font-medium transition-all"
          >
            <Plus className="w-4 h-4" />
            <span>New Chat / Clear History</span>
          </button>
        )}

        {/* Groq API Key Section */}
        <div className="space-y-2">
          <div className="flex items-center justify-between px-1">
            <span className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider">Groq API Configuration</span>
            {keySaved ? (
              <span className="text-[9px] text-emerald-400 font-semibold flex items-center">
                <Check className="w-2.5 h-2.5 mr-0.5" /> Configured
              </span>
            ) : (
              <span className="text-[9px] text-amber-500 font-semibold flex items-center">
                <AlertCircle className="w-2.5 h-2.5 mr-0.5" /> Missing Key
              </span>
            )}
          </div>
          <div className="relative">
            <Key className="w-3.5 h-3.5 text-zinc-500 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type={showKey ? "text" : "password"}
              value={apiKey}
              onChange={handleSaveKey}
              placeholder="Enter Groq API Key..."
              className="w-full bg-chat-bg border border-chat-border focus:border-blue-500 rounded-lg py-2 pl-9 pr-9 text-xs text-zinc-200 placeholder-zinc-600 outline-none transition-all"
            />
            <button
              onClick={() => setShowKey(!showKey)}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-zinc-500 hover:text-zinc-300"
            >
              {showKey ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
            </button>
          </div>
          <p className="text-[9px] text-zinc-500 px-1 leading-normal">
            API key is stored locally in your browser and used only for RAG requests.
          </p>
        </div>

        {/* Document Upload section */}
        <div className="space-y-2">
          <span className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider px-1">PDF Upload</span>
          <UploadSection onUploadSuccess={onUploadSuccess} onProcessSuccess={onProcessSuccess} />
        </div>

        {/* Indexed PDFs list */}
        <div className="space-y-2 flex-grow flex flex-col min-h-0">
          <span className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider px-1">Indexed PDFs</span>
          {documents.length === 0 ? (
            <div className="text-[11px] text-zinc-600 text-center py-4 border border-dashed border-chat-border rounded-lg">
              No PDFs uploaded or indexed
            </div>
          ) : (
            <div className="space-y-1.5 max-h-56 overflow-y-auto pr-1">
              {documents.map((doc, idx) => (
                <div
                  key={idx}
                  className="flex items-center justify-between p-2 rounded-lg bg-chat-card/30 border border-chat-border hover:bg-chat-card/50 transition-colors group min-w-0"
                >
                  <div className="flex items-center space-x-2 min-w-0 flex-1">
                    <FileText className={`w-3.5 h-3.5 flex-shrink-0 ${doc.is_indexed ? 'text-blue-400' : 'text-zinc-500'}`} />
                    <div className="min-w-0 flex-1">
                      <p className="text-xs text-zinc-300 truncate" title={doc.filename}>{doc.filename}</p>
                      <span className="text-[9px] text-zinc-500">{doc.size_kb} KB • {doc.is_indexed ? 'Indexed' : 'Uploaded'}</span>
                    </div>
                  </div>
                  <button
                    disabled={isDeleting === doc.filename}
                    onClick={() => handleDeleteDoc(doc.filename)}
                    className="text-zinc-500 hover:text-red-400 p-1 rounded transition-colors"
                    title="Delete document"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Sidebar;
