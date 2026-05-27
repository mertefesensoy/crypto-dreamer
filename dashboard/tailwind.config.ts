import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      fontFamily: {
        mono: [
          "ui-monospace",
          "SFMono-Regular",
          "Menlo",
          "Monaco",
          "Consolas",
          "monospace",
        ],
      },
      colors: {
        // Action color palette for chart consistency.
        action: {
          0: "#71717a", // 0% — neutral / cash
          1: "#3f6212", // 25%
          2: "#65a30d", // 50%
          3: "#a3e635", // 75%
          4: "#d9f99d", // 100% — full long
        },
      },
    },
  },
  plugins: [],
};

export default config;
