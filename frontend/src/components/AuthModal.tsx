import React, { useState } from 'react';
import { X, Lock, Mail, User, AlertCircle, ArrowRight, Loader2, Sparkles } from 'lucide-react';
import { useAuth } from '../context/AuthContext';

interface AuthModalProps {
  isOpen: boolean;
  onClose: () => void;
  defaultMode?: 'login' | 'register';
}

export const AuthModal: React.FC<AuthModalProps> = ({
  isOpen,
  onClose,
  defaultMode = 'login'
}) => {
  const { login, register } = useAuth();
  const [isLoginMode, setIsLoginMode] = useState(defaultMode === 'login');
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  if (!isOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);

    try {
      if (isLoginMode) {
        await login(username || email, password);
      } else {
        if (!email.trim() || !username.trim() || !password.trim()) {
          throw new Error('Please fill in all fields.');
        }
        await register(username, email, password);
      }
      onClose();
      // Reset form
      setUsername('');
      setEmail('');
      setPassword('');
    } catch (err: any) {
      setError(err.message || 'Authentication failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const switchMode = () => {
    setIsLoginMode(!isLoginMode);
    setError(null);
  };

  return (
    <div 
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-fade-in"
      onClick={onClose}
    >
      <div 
        className="relative w-full max-w-md bg-white border border-stone-200 rounded-2xl shadow-2xl p-6 sm:p-8 text-stone-900"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Close Button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 p-2 text-stone-400 hover:text-stone-700 rounded-lg hover:bg-stone-100 transition cursor-pointer"
          aria-label="Close"
        >
          <X className="w-5 h-5" />
        </button>

        {/* Header */}
        <div className="flex items-center gap-3 mb-6">
          <div className="p-2.5 rounded-xl bg-emerald-50 text-emerald-800 border border-emerald-200">
            <Sparkles className="w-6 h-6" />
          </div>
          <div>
            <h2 className="text-2xl font-bold tracking-tight text-stone-950">
              {isLoginMode ? 'Login' : 'Create an Account'}
            </h2>
          </div>
        </div>

        {/* Error message */}
        {error && (
          <div className="mb-4 p-3 bg-rose-50 border border-rose-200 text-rose-800 rounded-xl text-xs flex items-center gap-2">
            <AlertCircle className="w-4 h-4 shrink-0 text-rose-600" />
            <span>{error}</span>
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-xs font-bold uppercase tracking-wider text-stone-700 mb-1.5">
              {isLoginMode ? 'Username or Email' : 'Username'}
            </label>
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-stone-400">
                <User className="w-4 h-4" />
              </div>
              <input
                type="text"
                required
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder={isLoginMode ? 'Enter username or email' : 'Choose a username'}
                className="w-full pl-10 pr-4 py-2.5 bg-stone-50 hover:bg-white focus:bg-white border border-stone-300 rounded-xl text-sm focus:outline-none focus:border-emerald-700 focus:ring-1 focus:ring-emerald-700 text-stone-900 placeholder-stone-400 transition"
              />
            </div>
          </div>

          {!isLoginMode && (
            <div>
              <label className="block text-xs font-bold uppercase tracking-wider text-stone-700 mb-1.5">
                Email Address
              </label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-stone-400">
                  <Mail className="w-4 h-4" />
                </div>
                <input
                  type="email"
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="your.email@example.com"
                  className="w-full pl-10 pr-4 py-2.5 bg-stone-50 hover:bg-white focus:bg-white border border-stone-300 rounded-xl text-sm focus:outline-none focus:border-emerald-700 focus:ring-1 focus:ring-emerald-700 text-stone-900 placeholder-stone-400 transition"
                />
              </div>
            </div>
          )}

          <div>
            <label className="block text-xs font-bold uppercase tracking-wider text-stone-700 mb-1.5">
              Password
            </label>
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-stone-400">
                <Lock className="w-4 h-4" />
              </div>
              <input
                type="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                className="w-full pl-10 pr-4 py-2.5 bg-stone-50 hover:bg-white focus:bg-white border border-stone-300 rounded-xl text-sm focus:outline-none focus:border-emerald-700 focus:ring-1 focus:ring-emerald-700 text-stone-900 placeholder-stone-400 transition"
              />
            </div>
          </div>

          {/* Primary Action Button */}
          <button
            type="submit"
            disabled={loading}
            className="w-full mt-4 py-3 px-4 bg-emerald-700 hover:bg-emerald-800 active:bg-emerald-900 disabled:opacity-50 text-white font-bold rounded-xl text-sm flex items-center justify-center gap-2 shadow-md shadow-emerald-900/20 transition cursor-pointer"
          >
            {loading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin text-white" />
                <span>{isLoginMode ? 'Logging In...' : 'Creating Account...'}</span>
              </>
            ) : (
              <>
                <span className="text-white font-bold">{isLoginMode ? 'Login' : 'Create Account'}</span>
                <ArrowRight className="w-4 h-4 text-white" />
              </>
            )}
          </button>
        </form>

        {/* Switch Mode Footer */}
        <div className="mt-6 pt-4 border-t border-stone-200 text-center">
          <p className="text-xs text-stone-600">
            {isLoginMode ? "Don't have an account yet?" : 'Already have an account?'}{' '}
            <button
              type="button"
              onClick={switchMode}
              className="text-emerald-700 hover:text-emerald-800 font-bold underline underline-offset-2 ml-1 cursor-pointer"
            >
              {isLoginMode ? 'Register' : 'Login'}
            </button>
          </p>
        </div>
      </div>
    </div>
  );
};
