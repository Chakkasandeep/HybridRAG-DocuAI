import React, { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import ChatWindow from './components/ChatWindow';
import DashboardPage from './pages/DashboardPage';
import api from './utils/api';

function App() {
  const [activePage, setActivePage] = useState('chat');
  const [documents, setDocuments] = useState([]);
  const [messages, setMessages] = useState([]);
  const [chatLoading, setChatLoading] = useState(false);

  const fetchDocuments = async () => {
    try {
      const res = await api.get('/documents');
      setDocuments(res.data);
    } catch (err) {
      console.error("Error fetching documents:", err);
    }
  };

  const handleSendMessage = async (question) => {
    // 1. Add user message
    const userMsg = { role: 'user', content: question };
    setMessages((prev) => [...prev, userMsg]);
    setChatLoading(true);

    try {
      // 2. Query FastAPI backend
      const res = await api.post('/chat', { question });
      
      // 3. Add assistant message with metrics & citations
      const assistantMsg = {
        role: 'assistant',
        content: res.data.answer,
        metrics: res.data.metrics,
        sources: res.data.sources,
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err) {
      console.error("Error in chat request:", err);
      const errMsg = err.response?.data?.detail || "An unexpected error occurred. Please check your Groq API Key or backend server.";
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: `⚠️ Error: ${errMsg}`,
          error: true,
        },
      ]);
    } finally {
      setChatLoading(false);
    }
  };

  const handleClearChat = () => {
    setMessages([]);
  };

  useEffect(() => {
    fetchDocuments();
  }, []);

  const indexExists = documents.some(doc => doc.is_indexed);

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-chat-bg text-zinc-100 font-sans">
      {/* Sidebar */}
      <Sidebar
        activePage={activePage}
        setActivePage={setActivePage}
        documents={documents}
        setDocuments={setDocuments}
        onClearChat={handleClearChat}
        onUploadSuccess={fetchDocuments}
        onProcessSuccess={fetchDocuments}
        fetchDocuments={fetchDocuments}
      />

      {/* Main Area */}
      <main className="flex-1 flex flex-col h-full min-w-0 overflow-hidden">
        {activePage === 'chat' ? (
          <ChatWindow
            messages={messages}
            onSendMessage={handleSendMessage}
            loading={chatLoading}
            indexExists={indexExists}
          />
        ) : (
          <DashboardPage />
        )}
      </main>
    </div>
  );
}

export default App;
