import React, { useState, useEffect } from 'react';
import { Play, RefreshCw, BarChart3, HelpCircle } from 'lucide-react';
import EvaluationTable from '../components/EvaluationTable';
import MetricChart from '../components/MetricChart';
import api from '../utils/api';

const DashboardPage = () => {
  const [results, setResults] = useState({});
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');

  const fetchResults = async () => {
    try {
      const res = await api.get('/evaluation/results');
      setResults(res.data);
      setErrorMsg('');
    } catch (err) {
      // It's expected to fail if no evaluation has run yet
      setResults({});
    }
  };

  const handleRunEvaluation = async () => {
    setLoading(true);
    setErrorMsg('');
    try {
      const res = await api.post('/evaluate');
      setResults(res.data);
    } catch (err) {
      console.error(err);
      setErrorMsg(err.response?.data?.detail || 'Failed to execute evaluation benchmarks.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchResults();
  }, []);

  const hasData = Object.keys(results).length > 0;

  const configs = [
    { key: 'Semantic_Only', name: 'Semantic Search' },
    { key: 'Keyword_Only', name: 'Keyword Search (BM25)' },
    { key: 'Hybrid_Only', name: 'Hybrid Search' },
    { key: 'Hybrid_Reranked', name: 'Hybrid + Reranker' }
  ];

  let recommendedMethod = "Hybrid + Reranker";
  let fastestMethod = "Keyword (BM25)";

  if (hasData) {
    let bestConfig = configs[0];
    let maxPrecision = -1;
    let minLatencyForBest = Infinity;

    configs.forEach((c) => {
      const metrics = results[c.key] || {};
      const precision = metrics['precision@5'] || 0;
      const latency = metrics['retrieval_latency_ms'] || Infinity;

      if (precision > maxPrecision) {
        maxPrecision = precision;
        minLatencyForBest = latency;
        bestConfig = c;
      } else if (precision === maxPrecision && latency < minLatencyForBest) {
        minLatencyForBest = latency;
        bestConfig = c;
      }
    });
    recommendedMethod = bestConfig.name;

    let fastestConfig = configs[0];
    let minLatency = Infinity;

    configs.forEach((c) => {
      const metrics = results[c.key] || {};
      const latency = metrics['retrieval_latency_ms'] || Infinity;

      if (latency < minLatency) {
        minLatency = latency;
        fastestConfig = c;
      }
    });
    fastestMethod = fastestConfig.name;
  }

  return (
    <div className="flex-1 flex flex-col h-full overflow-hidden bg-chat-bg">
      {/* Header */}
      <header className="h-14 border-b border-chat-border flex items-center justify-between px-6 bg-chat-bg/85 backdrop-blur z-10 flex-shrink-0">
        <h2 className="text-sm font-semibold text-zinc-100 flex items-center space-x-2">
          <BarChart3 className="w-4 h-4 text-blue-400" />
          <span>RAG Evaluation Dashboard</span>
        </h2>
        <button
          onClick={handleRunEvaluation}
          disabled={loading}
          className="bg-blue-600 hover:bg-blue-700 disabled:bg-blue-800 disabled:opacity-50 text-white text-xs font-semibold py-2 px-4 rounded-lg flex items-center space-x-1.5 transition-colors"
        >
          {loading ? (
            <RefreshCw className="w-3.5 h-3.5 animate-spin" />
          ) : (
            <Play className="w-3.5 h-3.5" />
          )}
          <span>{loading ? 'Evaluating Suite...' : 'Run Evaluation Suite'}</span>
        </button>
      </header>

      {/* Main content scroll */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        <div className="max-w-4xl mx-auto space-y-6">
          {/* Top Info Alert */}
          <div className="bg-chat-sidebar border border-chat-border rounded-xl p-4 flex items-start space-x-3">
            <HelpCircle className="w-5 h-5 text-blue-400 mt-0.5 flex-shrink-0" />
            <div className="text-xs text-zinc-400 leading-relaxed">
              <span className="font-semibold text-zinc-200 block mb-0.5">About the Evaluation Benchmark</span>
              This suite tests the retrieval performance of four pipeline configurations (Semantic Search, Keyword BM25, Hybrid, and Hybrid + Reranker) over a standard set of evaluation QA pairs, injecting distractor documents to evaluate retrieval precision and latency.
            </div>
          </div>

          {errorMsg && (
            <div className="p-3 bg-red-950/30 text-red-400 border border-red-900/50 rounded-xl text-xs">
              {errorMsg}
            </div>
          )}

          {loading ? (
            <div className="flex flex-col items-center justify-center py-20 space-y-4">
              <RefreshCw className="w-10 h-10 text-blue-500 animate-spin" />
              <div className="text-center">
                <p className="text-sm font-semibold text-zinc-200">Benchmarking Retrieval Setup</p>
                <p className="text-xs text-zinc-500 mt-1">This will index evaluation documents and test accuracy metrics...</p>
              </div>
            </div>
          ) : !hasData ? (
            <div className="border border-dashed border-chat-border rounded-2xl py-24 text-center px-4">
              <BarChart3 className="w-12 h-12 text-zinc-600 mx-auto mb-3" />
              <h3 className="text-zinc-300 font-bold text-sm">No Evaluation Data Generated</h3>
              <p className="text-zinc-500 text-xs mt-1 max-w-sm mx-auto leading-relaxed">
                Run the evaluation suite by clicking the button in the top right. This will benchmark the pipeline's precision and speed.
              </p>
            </div>
          ) : (
            <div className="space-y-6 animate-fadeIn">
              {/* Comparative Table */}
              <div className="space-y-2">
                <h3 className="text-xs font-bold text-zinc-400 uppercase tracking-wider px-1">Comparative Performance Table</h3>
                <EvaluationTable data={results} />
              </div>

              {/* Insights */}
              <div className="bg-[#101525] border border-chat-border rounded-xl p-4 grid grid-cols-2 gap-4">
                <div className="text-center py-1">
                  <span className="text-[10px] text-zinc-500 uppercase tracking-wider font-semibold">Recommended Method</span>
                  <p className="text-lg font-bold text-emerald-400 mt-1">{recommendedMethod}</p>
                </div>
                <div className="text-center py-1 border-l border-chat-border">
                  <span className="text-[10px] text-zinc-500 uppercase tracking-wider font-semibold">Fastest Method</span>
                  <p className="text-lg font-bold text-amber-500 mt-1">{fastestMethod}</p>
                </div>
              </div>

              {/* Dynamic Charts Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <MetricChart data={results} type="precision" />
                <MetricChart data={results} type="latency" />
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default DashboardPage;
