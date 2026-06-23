import { useState } from "react";
import SessionList from "./SessionList";
import AnnotateEditor from "./AnnotateEditor";

// ponytail: no router lib — 2 routes, pathname switch is enough.
export default function App() {
  const [path] = useState(() => window.location.pathname);

  const match = path.match(/^\/ui\/([^/]+)\/?$/);
  if (match) {
    return <AnnotateEditor sessionId={decodeURIComponent(match[1])} />;
  }
  return <SessionList />;
}