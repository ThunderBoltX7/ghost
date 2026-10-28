#!/usr/bin/env node

const { spawn, spawnSync } = require("child_process");
const path = require("path");

function getPythonCommand() {
  const commands = ["python3", "python", "py"];
  for (const cmd of commands) {
    const res = spawnSync(cmd, ["--version"], { stdio: "ignore" });
    if (res.status === 0) return cmd;
  }
  return null;
}

const pyCmd = getPythonCommand();
if (!pyCmd) {
  console.error("\x1b[31mError: Python 3 is required but was not found on your PATH.\x1b[0m");
  console.error("Please install Python (https://www.python.org/) and try again.");
  process.exit(1);
}

// Forward all CLI arguments to the ghost_cli module
const child = spawn(pyCmd, ["-m", "ghost_cli.main", ...process.argv.slice(2)], {
  stdio: "inherit",
  cwd: process.cwd(),
});

child.on("exit", (code) => {
  process.exit(code ?? 0);
});