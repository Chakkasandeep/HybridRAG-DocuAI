import React, { useState } from 'react';
import { Bot, User, BarChart2, BookOpen, ChevronDown, ChevronUp } from 'lucide-react';
import CitationCard from './CitationCard';

const MessageBubble = ({ role, content, metrics, sources }) => {
  const isAssistant = role === 'assistant';
  const [showMetrics, setShowMetrics] = useState(false);
  const [showSources, setShowSources] = useState(false);

  return (
    <div className={`flex w-full py-6 justify-center ${isAssistant ? 'bg-[#0f1422] border-y border-chat-border' : 'bg-transparent'}`}>
      <div className="flex w-full max-w-4xl px-4 space-x-4">
        {/* Avatar */}
        <div className={`w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 ${
          isAssistant ? 'bg-blue-600/10 text-blue-400 border border-blue-500/20' : 'bg-zinc-700/20 text-zinc-300 border border-zinc-700/50'
        }`}>
          {isAssistant ? <Bot className="w-5 h-5" /> : <User className="w-5 h-5" />}
        </div>

        {/* Content Area */}
        <div className="flex-1 space-y-4 min-w-0">
          <div className="text-sm font-semibold text-zinc-300">
            {isAssistant ? 'AI Assistant' : 'You'}
          </div>

          <div className="text-zinc-100 text-sm leading-relaxed whitespace-pre-wrap select-text selection:bg-blue-500/30">
            {content}
          </div>

          {isAssistant && (metrics || sources) && (
            <div className="space-y-3 pt-2">
              {/* Expanders header */}
              <div className="flex items-center space-x-2 text-xs">
                {metrics && (
                  <button
                    onClick={() => setShowMetrics(!showMetrics)}
                    className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg border transition-colors ${
                      showMetrics 
                        ? 'bg-blue-500/10 border-blue-500/30 text-blue-400' 
                        : 'bg-chat-card border-chat-border text-zinc-400 hover:text-zinc-200'
                    }`}
                  >
                    <BarChart2 className="w-3.5 h-3.5" />
                    <span>Performance Metrics</span>
                    {showMetrics ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                  </button>
                )}

                {sources && sources.length > 0 && (
                  <button
                    onClick={() => setShowSources(!showSources)}
                    className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg border transition-colors ${
                      showSources 
                        ? 'bg-blue-500/10 border-blue-500/30 text-blue-400' 
                        : 'bg-chat-card border-chat-border text-zinc-400 hover:text-zinc-200'
                    }`}
                  >
                    <BookOpen className="w-3.5 h-3.5" />
                    <span>Sources & Citations</span>
                    {showSources ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                  </button>
                )}
              </div>

              {/* Performance Metrics Panel */}
              {showMetrics && metrics && (
                <div className="bg-chat-card border border-chat-border rounded-lg p-4 grid grid-cols-2 sm:grid-cols-4 gap-4 animate-fadeIn">
                  <div className="space-y-0.5">
                    <span className="text-[10px] text-zinc-500 uppercase tracking-wider font-semibold">Retrieval Time</span>
                    <p className="text-sm font-bold text-zinc-200">{(metrics.retrieval_latency * 1000).toFixed(1)} ms</p>
                  </div>
                  <div className="space-y-0.5">
                    <span className="text-[10px] text-zinc-500 uppercase tracking-wider font-semibold">Rerank Time</span>
                    <p className="text-sm font-bold text-zinc-200">{(metrics.rerank_latency * 1000).toFixed(1)} ms</p>
                  </div>
                  <div className="space-y-0.5">
                    <span className="text-[10px] text-zinc-500 uppercase tracking-wider font-semibold">Generation Time</span>
                    <p className="text-sm font-bold text-zinc-200">{metrics.generation_latency.toFixed(2)} s</p>
                  </div>
                  <div className="space-y-0.5">
                    <span className="text-[10px] text-zinc-500 uppercase tracking-wider font-semibold">Chunks (Retr. / Used)</span>
                    <p className="text-sm font-bold text-zinc-200">{metrics.num_candidates} / {metrics.num_reranked}</p>
                  </div>
                </div>
              )}

              {/* Sources List Panel */}
              {showSources && sources && sources.length > 0 && (
                <div className="space-y-2 animate-fadeIn">
                  {sources.map((src, idx) => (
                    <CitationCard
                      key={src.id || idx}
                      index={idx + 1}
                      source={src.source}
                      page={src.page}
                      fusionScore={src.fusion_score}
                      rerankConfidence={src.rerank_confidence}
                      content={src.content}
                    />
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default MessageBubble;
