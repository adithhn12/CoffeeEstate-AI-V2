/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        estate: {
          50: '#f2f8f5',
          100: '#e1efe8',
          200: '#c5dfd4',
          300: '#9dc7b7',
          400: '#6fa995',
          500: '#4d8e79',
          600: '#3a7260',
          700: '#2f5b4e',
          800: '#28493f',
          900: '#1b4332',  // Primary Deep Forest Green
          950: '#0f271d',
        },
        sage: {
          50: '#f4f7f5',
          100: '#e5ece7',
          200: '#cddbd2',
          300: '#a8c2b3',
          400: '#7ea490',
          500: '#52b788',  // Muted Sage Accent
          600: '#46725b',
          700: '#3a5b4a',
        },
        amberGold: {
          50: '#fdf9f0',
          100: '#f9f0db',
          200: '#f2ddb3',
          300: '#eac585',
          400: '#e2ac5a',
          500: '#d4a373',  // Restrained Amber Highlight
          600: '#b87c32',
          700: '#925c27',
        },
        warmBg: '#fcfbfa',
        cardBg: '#ffffff',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
