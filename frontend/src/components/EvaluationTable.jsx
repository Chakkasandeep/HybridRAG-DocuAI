import React from 'react';

const EvaluationTable = ({ data }) => {
  const configs = [
    { key: 'Semantic_Only', name: 'Semantic Search' },
    { key: 'Keyword_Only', name: 'Keyword Search (BM25)' },
    { key: 'Hybrid_Only', name: 'Hybrid Search' },
    { key: 'Hybrid_Reranked', name: 'Hybrid + Reranker' }
  ];

  return (
    <div className="overflow-hidden border border-chat-border rounded-xl bg-chat-sidebar">
      <table className="w-full text-left border-collapse text-xs">
        <thead>
          <tr className="bg-chat-card text-zinc-400 font-semibold border-b border-chat-border">
            <th className="py-3 px-5">Method</th>
            <th className="py-3 px-5 text-center">Precision@5</th>
            <th className="py-3 px-5 text-center">Recall@5</th>
            <th className="py-3 px-5 text-center">Latency (ms)</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-chat-border text-zinc-300">
          {configs.map((config) => {
            const metrics = data[config.key] || {};
            const precision = metrics['precision@5'] !== undefined ? metrics['precision@5'].toFixed(3) : 'N/A';
            const recall = metrics['recall@5'] !== undefined ? metrics['recall@5'].toFixed(3) : 'N/A';
            const latency = metrics['retrieval_latency_ms'] !== undefined ? `${metrics['retrieval_latency_ms'].toFixed(1)} ms` : 'N/A';
            
            // Highlight the hybrid reranked row as the best performing configuration
            const isBest = config.key === 'Hybrid_Reranked';

            return (
              <tr 
                key={config.key} 
                className={`hover:bg-chat-card/20 transition-colors ${isBest ? 'bg-blue-600/5 font-medium text-zinc-100' : ''}`}
              >
                <td className="py-3 px-5 flex items-center space-x-2">
                  <span className={`w-1.5 h-1.5 rounded-full ${
                    config.key === 'Semantic_Only' ? 'bg-purple-500' :
                    config.key === 'Keyword_Only' ? 'bg-amber-500' :
                    config.key === 'Hybrid_Only' ? 'bg-cyan-500' : 'bg-emerald-500'
                  }`} />
                  <span>{config.name}</span>
                </td>
                <td className="py-3 px-5 text-center font-mono">{precision}</td>
                <td className="py-3 px-5 text-center font-mono">{recall}</td>
                <td className="py-3 px-5 text-center font-mono text-zinc-400">{latency}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
};

export default EvaluationTable;
