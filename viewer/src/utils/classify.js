export const READONLY_TOOLS = [
  "read", "readfile", "read_file", "view", "cat", "glob", "grep", "ls",
  "list", "search", "codebase_search", "web_search", "websearch", "webfetch",
  "web_fetch", "fetch", "conversation_search", "recent_chats", "get", "find"
];
export const MUTATING_TOOLS = [
  "write", "writefile", "write_file", "edit", "str_replace", "create_file",
  "createfile", "multiedit", "apply_patch", "patch", "delete", "remove",
  "mkdir", "move", "rename"
];
export const PLANNING_TOOLS = ["todowrite", "todoread", "todo", "update_plan", "plan", "planmode"];
export const SUBAGENT_TOOLS = ["task", "agent", "dispatch_agent", "subagent", "spawn", "delegate"];

export const MUTATING_SHELL = /\b(mkdir|rmdir|rm\s|mv\s|cp\s|touch\s|tee\s|chmod|chown|ln\s|git\s+(commit|push|add|checkout|reset|merge|rebase)|pip3?\s+install|npm\s+(install|i)\b|yarn\s+add|apt(-get)?\s+install|>{1,2}\s*[^&|]|json\.dump|\.write\(|open\([^)]*['"]w)/;

export function classifyTool(name, args) {
  const n = String(name || "").toLowerCase();
  if (SUBAGENT_TOOLS.indexOf(n) !== -1 || /(^|_)agent($|_)/.test(n)) return "subagent";
  for (const k of PLANNING_TOOLS) if (n === k || n.indexOf(k) === 0) return "planning";
  for (const k of MUTATING_TOOLS) if (n === k || n.indexOf(k) === 0) return "mutating";
  if (n === "bash" || n === "shell" || n === "terminal" || n === "exec" || n === "run" || n === "cmd") {
    const cmd = args ? String(args.command || args.cmd || args.script || "") : "";
    return MUTATING_SHELL.test(cmd) ? "mutating" : "readonly";
  }
  for (const k of READONLY_TOOLS) if (n === k || n.indexOf(k) === 0) return "readonly";
  return "other";
}

export const DEAD_RE = /command not found|no such file or directory|traceback \(most recent|is not recognized as|permission denied|fatal:|\bENOENT\b|zsh:\s*(\d+:\s*)?(command not found|no matches found|parse error)|syntaxerror|\berror\b[:\s]/i;

export function isDeadEnd(resultText) {
  if (!resultText) return false;
  return DEAD_RE.test(String(resultText).slice(0, 6000));
}
