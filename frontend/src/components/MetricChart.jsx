import React from 'react';

const MetricChart = ({ data, type }) => {
  const configs = [
    { key: 'Semantic_Only', name: 'Semantic Search' },
    { key: 'Keyword_Only', name: 'Keyword (BM25)' },
    { key: 'Hybrid_Only', name: 'Hybrid Search' },
    { key: 'Hybrid_Reranked', name: 'Hybrid + Reranker' }
  ];

  const colors = [
    'bg-purple-500', 
    'bg-amber-500', 
    'bg-cyan-500', 
    'bg-emerald-500'
  ];

  const isPrecision = type === 'precision';

  // Find max latency to calculate relative bar widths for latency chart
  const maxLatency = Math.max(
    ...configs.map(c => data[c.key]?.['retrieval_latency_ms'] || 1),
    1
  );

  return (
    <div className="bg-chat-sidebar border border-chat-border rounded-xl p-5 flex flex-col min-w-0">
      <h3 className="text-xs font-bold text-zinc-400 uppercase tracking-wider mb-5">
        {isPrecision ? '1. Precision@5 Comparison' : '2. Retrieval Latency (ms) Comparison'}
      </h3>
      <div className="space-y-4">
        {configs.map((config, idx) => {
          const metrics = data[config.key] || {};
          const value = isPrecision 
            ? (metrics['precision@5'] || 0) 
            : (metrics['retrieval_latency_ms'] || 0);

          // Calculate percentage width for the bar
          const widthPct = isPrecision 
            ? `${value * 100}%` 
            : `${(value / maxLatency) * 100}%`;

          const formattedValue = isPrecision 
            ? value.toFixed(3) 
            : `${value.toFixed(1)} ms`;

          return (
            <div key={config.key} className="space-y-1.5">
              <div className="flex justify-between text-xs">
                <span className="text-zinc-300 font-medium">{config.name}</span>
                <span className="text-zinc-100 font-bold font-mono">{formattedValue}</span>
              </div>
              <div className="w-full bg-chat-bg rounded-full h-2.5 overflow-hidden border border-chat-border/30">
                <div 
                  className={`h-full ${colors[idx]} rounded-full transition-all duration-500`}
                  style={{ width: widthPct }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default MetricChart;
