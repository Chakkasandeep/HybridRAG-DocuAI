import React, { useState, useRef, useEffect } from 'react';
import { Send, RefreshCw, Bot } from 'lucide-react';
import MessageBubble from './MessageBubble';

const ChatWindow = ({ messages, onSendMessage, loading, indexExists }) => {
  const [question, setQuestion] = useState('');
  const messagesEndRef = useRef(null);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (question.trim() && !loading && indexExists) {
      onSendMessage(question);
      setQuestion('');
    }
  };

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  return (
    <div className="flex-1 flex flex-col h-full overflow-hidden bg-chat-bg relative">
      {/* Header */}
      <header className="h-14 border-b border-chat-border flex items-center justify-between px-6 bg-chat-bg/85 backdrop-blur z-10 flex-shrink-0">
        <h2 className="text-sm font-semibold text-zinc-100 flex items-center space-x-2">
          <span>Chat with PDF</span>
          <span className="bg-blue-600/10 text-blue-400 text-[10px] px-1.5 py-0.5 rounded font-mono">Llama 3.3 70B</span>
        </h2>
      </header>

      {/* Chat Messages Feed */}
      <div className="flex-1 overflow-y-auto overflow-x-hidden flex flex-col">
        {messages.length === 0 ? (
          <div className="flex-grow flex flex-col items-center justify-center text-center p-8 space-y-4 max-w-lg mx-auto">
            <div className="bg-blue-600/10 text-blue-400 p-4 rounded-2xl border border-blue-500/20">
              <Bot className="w-12 h-12" />
            </div>
            <h3 className="text-zinc-200 font-bold text-lg">Ask questions from your Documents</h3>
            <p className="text-zinc-400 text-xs leading-relaxed">
              {indexExists 
                ? "Upload and index your PDFs in the sidebar, then ask questions. DocuAI will use Hybrid Retrieval (FAISS + BM25) and Cross-Encoder reranking to find answers with precise citations."
                : "Please upload and process at least one PDF file in the sidebar to get started."}
            </p>
          </div>
        ) : (
          <div className="flex flex-col">
            {messages.map((msg, index) => (
              <MessageBubble
                key={index}
                role={msg.role}
                content={msg.content}
                metrics={msg.metrics}
                sources={msg.sources}
              />
            ))}
          </div>
        )}

        {/* Loading Bubble */}
        {loading && (
          <div className="flex w-full py-6 justify-center bg-[#0f1422] border-y border-chat-border animate-pulse">
            <div className="flex w-full max-w-4xl px-4 space-x-4">
              <div className="w-8 h-8 rounded-lg bg-blue-600/10 text-blue-400 border border-blue-500/20 flex items-center justify-center flex-shrink-0">
                <Bot className="w-5 h-5" />
              </div>
              <div className="flex-1 space-y-3">
                <div className="text-sm font-semibold text-zinc-400">AI Assistant</div>
                <div className="flex items-center space-x-1.5 text-zinc-400 text-xs">
                  <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                  <span>Synthesizing answer using Llama 3.3 70B...</span>
                </div>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Panel */}
      <div className="p-4 bg-chat-bg/90 border-t border-chat-border flex-shrink-0">
        <form onSubmit={handleSubmit} className="max-w-4xl mx-auto relative">
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder={indexExists ? "Ask a question from the PDF files..." : "Upload and index documents to start chatting"}
            disabled={loading || !indexExists}
            className="w-full bg-chat-sidebar border border-chat-border focus:border-blue-500 focus:ring-1 focus:ring-blue-500 rounded-xl py-3.5 pl-4 pr-12 text-sm text-zinc-100 placeholder-zinc-500 outline-none transition-all disabled:opacity-50 disabled:cursor-not-allowed"
          />
          <button
            type="submit"
            disabled={loading || !question.trim() || !indexExists}
            className="absolute right-2.5 top-1/2 -translate-y-1/2 bg-blue-600 hover:bg-blue-700 disabled:bg-zinc-800 disabled:text-zinc-600 text-white p-2 rounded-lg transition-colors"
          >
            <Send className="w-4 h-4" />
          </button>
        </form>
        <p className="text-[10px] text-center text-zinc-500 mt-2">
          DocuAI uses Hybrid Search & CrossEncoder reranking. Answers are strictly based on the facts in the uploaded context.
        </p>
      </div>
    </div>
  );
};

export default ChatWindow;
