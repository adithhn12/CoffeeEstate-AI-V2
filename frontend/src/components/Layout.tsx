import React, { useState, useEffect } from 'react';
import { NavLink, Outlet, useLocation, useNavigate } from 'react-router-dom';
import {
  ScanEye,
  ShieldCheck,
  ChevronRight,
  ChevronLeft,
  Sparkles,
  LogIn,
  LogOut,
  User as UserIcon,
  CheckCircle2
} from 'lucide-react';
import { api } from '../api/client';
import { HealthStatus, ModelStatusInfo } from '../types';
import { useAuth } from '../context/AuthContext';
import { AuthModal } from './AuthModal';

export const Layout: React.FC = () => {
  const [collapsed, setCollapsed] = useState(false);
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [modelStatus, setModelStatus] = useState<ModelStatusInfo | null>(null);
  const [loadingSeed, setLoadingSeed] = useState(false);
  const [authModalOpen, setAuthModalOpen] = useState(false);
  const [authModalMode, setAuthModalMode] = useState<'login' | 'register'>('login');

  const { user, logout } = useAuth();
  const location = useLocation();
  const navigate = useNavigate();

  useEffect(() => {
    fetchHealthAndModel();
    const timer = setInterval(fetchHealthAndModel, 30000);
    return () => clearInterval(timer);
  }, []);

  const fetchHealthAndModel = async () => {
    try {
      const [h, m] = await Promise.all([
        api.getHealth().catch(() => null),
        api.getModelStatus().catch(() => null),
      ]);
      setHealth(h);
      setModelStatus(m);
    } catch {
      // Graceful offline fallback
    }
  };

  const handleSeedData = async () => {
    try {
      setLoadingSeed(true);
      await api.seedSampleData();
      window.location.reload();
    } catch (err: any) {
      alert(`Could not load sample records: ${err.message}`);
    } finally {
      setLoadingSeed(false);
    }
  };

  const openAuth = (mode: 'login' | 'register') => {
    setAuthModalMode(mode);
    setAuthModalOpen(true);
  };

  const navItems = [
    {
      label: 'Ripeness Analysis',
      path: '/',
      altPath: '/analysis',
      icon: ScanEye,
      description: 'Cherry Detection & Maturity'
    },
    {
      label: 'Model Status',
      path: '/status',
      altPath: '/settings',
      icon: ShieldCheck,
      description: 'Diagnostics & Health'
    },
  ];

  const isNavActive = (item: typeof navItems[0]) => {
    if (item.path === '/') {
      return location.pathname === '/' || location.pathname === '/analysis';
    }
    return location.pathname.startsWith(item.path) || location.pathname.startsWith(item.altPath);
  };

  return (
    <div className="min-h-screen flex flex-col md:flex-row bg-[#faf8f5] text-stone-900">
      {/* ================= DESKTOP SIDEBAR ================= */}
      <aside
        className={`hidden md:flex flex-col bg-estate-950 text-white border-r border-estate-900 transition-all duration-300 z-30 shrink-0 ${
          collapsed ? 'w-20' : 'w-64'
        }`}
      >
        {/* Brand Header */}
        <div className="h-16 flex items-center justify-between px-4 border-b border-estate-900/80">
          {!collapsed && (
            <div className="flex items-center gap-2.5">
              <div className="w-9 h-9 rounded-xl bg-estate-800 flex items-center justify-center border border-estate-700">
                <span className="text-lg font-bold text-emerald-400">☕</span>
              </div>
              <div>
                <h1 className="font-bold text-sm tracking-tight text-white leading-tight">Coffee Estate AI</h1>
                <p className="text-[11px] text-estate-400">
                  Harvest & Ripeness System
                </p>
              </div>
            </div>
          )}
          {collapsed && (
            <div className="w-full flex justify-center">
              <span className="text-xl">☕</span>
            </div>
          )}
          <button
            onClick={() => setCollapsed(!collapsed)}
            className="p-1.5 rounded-lg text-estate-300 hover:text-white hover:bg-estate-900 transition-colors"
            title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            {collapsed ? <ChevronRight size={18} /> : <ChevronLeft size={18} />}
          </button>
        </div>

        {/* User Account Bar in Desktop Sidebar */}
        <div className="p-3 border-b border-estate-900/80 bg-estate-900/40">
          {user ? (
            <div className={`flex items-center ${collapsed ? 'justify-center' : 'justify-between'} gap-2`}>
              <div className="flex items-center gap-2.5 min-w-0">
                <div className="w-8 h-8 rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/40 flex items-center justify-center font-bold text-xs shrink-0">
                  {user.username.charAt(0).toUpperCase()}
                </div>
                {!collapsed && (
                  <div className="min-w-0">
                    <div className="flex items-center gap-1.5">
                      <span className="text-xs font-semibold text-white truncate max-w-[110px]">
                        {user.username}
                      </span>
                      <CheckCircle2 size={12} className="text-emerald-400 shrink-0" />
                    </div>
                    <span className="text-[10px] text-emerald-400/90 block">History Saving Active</span>
                  </div>
                )}
              </div>
              {!collapsed && (
                <button
                  onClick={logout}
                  title="Log Out"
                  className="p-1.5 text-estate-400 hover:text-rose-400 hover:bg-estate-800 rounded-lg transition-colors"
                >
                  <LogOut size={16} />
                </button>
              )}
            </div>
          ) : (
            <div className="flex flex-col gap-2">
              {!collapsed ? (
                <>
                  <div className="flex items-center justify-between text-[11px]">
                    <span className="text-estate-400">Mode</span>
                    <span className="px-1.5 py-0.5 rounded bg-estate-900 border border-estate-800 text-stone-300 font-medium">Guest</span>
                  </div>
                  <button
                    onClick={() => openAuth('login')}
                    className="w-full py-2.5 px-3 bg-estate-800 hover:bg-estate-700 active:bg-estate-900 border border-emerald-500/30 text-emerald-300 hover:text-white rounded-xl text-xs font-semibold flex items-center justify-center gap-1.5 shadow-sm transition"
                  >
                    <LogIn size={14} className="text-emerald-400" />
                    <span>Login</span>
                  </button>
                </>
              ) : (
                <button
                  onClick={() => openAuth('login')}
                  title="Login"
                  className="w-10 h-10 mx-auto rounded-xl bg-estate-800 border border-emerald-500/30 text-emerald-400 flex items-center justify-center hover:bg-estate-700 transition"
                >
                  <LogIn size={18} />
                </button>
              )}
            </div>
          )}
        </div>

        {/* Navigation List */}
        <nav className="flex-1 py-4 px-3 space-y-1.5 overflow-y-auto">
          {navItems.map((item) => {
            const Icon = item.icon;
            const active = isNavActive(item);

            return (
              <NavLink
                key={item.label}
                to={item.path}
                className={`flex items-center gap-3 px-3 py-3 rounded-xl text-sm font-medium transition-all ${
                  active
                    ? 'bg-estate-800 text-white shadow-xs font-semibold ring-1 ring-emerald-500/30'
                    : 'text-estate-200 hover:text-white hover:bg-estate-900/80'
                } ${collapsed ? 'justify-center' : ''}`}
                title={collapsed ? item.label : undefined}
              >
                <Icon size={20} className={active ? 'text-emerald-400' : 'text-estate-300'} />
                {!collapsed && (
                  <div className="min-w-0">
                    <span className="block leading-tight">{item.label}</span>
                    <span className="text-[11px] text-estate-400 font-normal block mt-0.5">
                      {item.description}
                    </span>
                  </div>
                )}
              </NavLink>
            );
          })}
        </nav>

        {/* Desktop Footer */}
        {!collapsed && (
          <div className="p-4 border-t border-estate-900/80 bg-estate-950/60 text-xs text-estate-300 space-y-2">
            <button
              onClick={handleSeedData}
              disabled={loadingSeed}
              className="w-full py-1.5 px-2 bg-estate-900/90 hover:bg-estate-800 text-estate-200 rounded-lg text-[11px] font-medium transition-colors flex items-center justify-center gap-1.5 border border-estate-800"
            >
              <Sparkles size={12} className="text-amber-400" />
              {loadingSeed ? 'Loading...' : 'Reset Demo Records'}
            </button>
          </div>
        )}
      </aside>

      {/* ================= MAIN CONTENT AREA ================= */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Mobile Top App Bar */}
        <header className="md:hidden sticky top-0 z-20 bg-white/95 backdrop-blur-md border-b border-stone-200/90 px-4 py-3 flex items-center justify-between shadow-2xs">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-xl bg-estate-900 text-white flex items-center justify-center font-bold text-sm shadow-xs">
              ☕
            </div>
            <div>
              <h1 className="text-sm font-bold text-stone-900 tracking-tight leading-tight">Coffee Estate AI</h1>
              <p className="text-[11px] text-stone-500">
                Harvest & Ripeness System
              </p>
            </div>
          </div>

          {/* Mobile Auth Button */}
          <div>
            {user ? (
              <div className="flex items-center gap-2">
                <span className="text-xs font-semibold text-stone-800 bg-stone-100 px-2 py-1 rounded-lg">
                  {user.username}
                </span>
                <button
                  onClick={logout}
                  className="p-1.5 text-stone-500 hover:text-rose-600 rounded-lg"
                  title="Log out"
                >
                  <LogOut size={16} />
                </button>
              </div>
            ) : (
              <button
                onClick={() => openAuth('login')}
                className="text-xs font-semibold px-3 py-1.5 bg-estate-900 hover:bg-estate-800 text-white rounded-xl shadow-xs transition"
              >
                Login
              </button>
            )}
          </div>
        </header>

        {/* Routed Page Content */}
        <main className="flex-1 px-4 py-5 sm:px-6 sm:py-6 md:p-8 max-w-5xl mx-auto w-full pb-24 md:pb-8">
          <Outlet />
        </main>
      </div>

      {/* ================= MOBILE BOTTOM NAVIGATION ================= */}
      <nav
        aria-label="Mobile Navigation"
        className="md:hidden fixed bottom-0 left-0 right-0 z-40 bg-white/98 backdrop-blur-lg border-t border-stone-200/90 shadow-lg px-3 py-2 flex items-center justify-around"
        style={{ paddingBottom: 'calc(0.5rem + env(safe-area-inset-bottom, 0px))' }}
      >
        {navItems.map((item) => {
          const Icon = item.icon;
          const active = isNavActive(item);

          return (
            <button
              key={item.label}
              onClick={() => navigate(item.path)}
              className={`flex-1 flex flex-col items-center justify-center py-2 px-3 rounded-xl transition-all min-h-[52px] touch-manipulation ${
                active ? 'text-estate-900 font-bold' : 'text-stone-500 hover:text-stone-800'
              }`}
            >
              <div
                className={`p-1.5 rounded-xl transition-transform ${
                  active ? 'bg-estate-100 text-estate-900 scale-105' : 'text-stone-500'
                }`}
              >
                <Icon size={22} strokeWidth={active ? 2.5 : 2} />
              </div>
              <span className={`text-[12px] mt-0.5 tracking-tight ${active ? 'font-bold text-estate-950' : 'font-medium'}`}>
                {item.label}
              </span>
            </button>
          );
        })}
      </nav>

      {/* Auth Modal */}
      <AuthModal
        isOpen={authModalOpen}
        onClose={() => setAuthModalOpen(false)}
        defaultMode={authModalMode}
      />
    </div>
  );
};
