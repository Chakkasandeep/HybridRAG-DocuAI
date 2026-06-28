import React, { useState } from 'react';
import { ChevronDown, ChevronUp } from 'lucide-react';

const CitationCard = ({ source, page, fusionScore, rerankConfidence, content, index }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div className="bg-chat-card border border-chat-border rounded-lg overflow-hidden transition-all duration-200">
      <div 
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center justify-between p-3.5 cursor-pointer hover:bg-chat-card/85 transition-colors select-none"
      >
        <div className="flex items-center space-x-2.5 min-w-0">
          <div className="bg-blue-500/10 text-blue-400 w-6 h-6 rounded flex items-center justify-center flex-shrink-0">
            <span className="text-xs font-bold font-mono">[{index}]</span>
          </div>
          <div className="min-w-0">
            <h4 className="text-xs font-semibold text-zinc-200 truncate max-w-[250px] sm:max-w-md" title={source}>
              {source}
            </h4>
            <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5 text-[10px] text-zinc-400 mt-0.5">
              <span>Page {page}</span>
              <span className="text-zinc-600">•</span>
              <span>Fusion Score: {fusionScore.toFixed(3)}</span>
              <span className="text-zinc-600">•</span>
              <span className="text-emerald-400 font-semibold">Relevance: {rerankConfidence}</span>
            </div>
          </div>
        </div>
        <div className="text-zinc-400 pl-2">
          {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        </div>
      </div>
      
      {isExpanded && (
        <div className="p-3.5 border-t border-chat-border bg-[#0e1322]">
          <p className="text-xs text-zinc-300 whitespace-pre-wrap leading-relaxed font-sans select-text selection:bg-blue-500/30">
            {content}
          </p>
        </div>
      )}
    </div>
  );
};

export default CitationCard;
