import { useState } from "react";
import PlateGallery from "./PlateGallery";
import SessionList from "./SessionList";
import AnnotateEditor from "./AnnotateEditor";

// ponytail: no router lib — 3 routes, pathname switch is enough.
export default function App() {
  const [path] = useState(() => window.location.pathname);

  const uiMatch = path.match(/^\/ui\/([^/]+)\/?$/);
  if (uiMatch) {
    return <AnnotateEditor sessionId={decodeURIComponent(uiMatch[1])} />;
  }
  if (path === "/sessions" || path === "/sessions/") {
    return <SessionList />;
  }
  return <PlateGallery />;
}
