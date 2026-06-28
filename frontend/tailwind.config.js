/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Dark theme color system resembling ChatGPT/Claude
        chat: {
          bg: '#0b0f19',        // Deep dark space background
          sidebar: '#0d111d',   // Accent sidebar background
          card: '#161b2c',      // Slightly lighter card/message background
          border: '#1f293d',    // Subtle borders
          userBubble: '#242b3d' // User chat bubble
        }
      }
    },
  },
  plugins: [],
}
