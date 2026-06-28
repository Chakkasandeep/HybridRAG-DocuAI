import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
});

// Interceptor to inject X-Groq-API-Key header if it exists in localStorage
api.interceptors.request.use((config) => {
  const apiKey = localStorage.getItem('groq_api_key');
  if (apiKey) {
    config.headers['X-Groq-API-Key'] = apiKey;
  }
  return config;
}, (error) => {
  return Promise.reject(error);
});

export default api;
export { API_BASE_URL };
