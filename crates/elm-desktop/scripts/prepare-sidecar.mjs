import { execFileSync } from "node:child_process";
import { copyFileSync, mkdirSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const scriptDirectory = dirname(fileURLToPath(import.meta.url));
const workspace = resolve(scriptDirectory, "../../..");
const extension = process.platform === "win32" ? ".exe" : "";
const target = execFileSync("rustc", ["--print", "host-tuple"], {
  cwd: workspace,
  encoding: "utf8",
}).trim();

execFileSync("cargo", ["build", "--release", "-p", "elm-daemon"], {
  cwd: workspace,
  stdio: "inherit",
});

const source = join(workspace, "target", "release", `elm-daemon${extension}`);
const binaryDirectory = join(workspace, "crates", "elm-desktop", "src-tauri", "binaries");
const destination = join(binaryDirectory, `elm-daemon-${target}${extension}`);
mkdirSync(binaryDirectory, { recursive: true });
copyFileSync(source, destination);
console.log(`Prepared ${destination}`);

