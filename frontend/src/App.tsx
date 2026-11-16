import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider } from './context/AuthContext';
import { Layout } from './components/Layout';
import { RipenessAnalysis } from './pages/RipenessAnalysis';
import { SettingsModel } from './pages/SettingsModel';

export const App: React.FC = () => {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Layout />}>
            <Route index element={<RipenessAnalysis />} />
            <Route path="analysis" element={<RipenessAnalysis />} />
            <Route path="status" element={<SettingsModel />} />
            <Route path="settings" element={<SettingsModel />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  );
};

export default App;
