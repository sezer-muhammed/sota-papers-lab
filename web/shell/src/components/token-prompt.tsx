import { useState } from "react";
import { KeyRound } from "lucide-react";
import { Button, Card, CardDescription, CardHeader, CardTitle } from "@spl/ui";
import { saveToken } from "../api";

export function TokenPrompt() {
  const [token, setToken] = useState("");
  return (
    <Card depth="deep" density="spacious" className="mx-auto max-w-md">
      <CardHeader>
        <CardTitle>This host needs a token</CardTitle>
        <CardDescription>
          Paste the value of SPL_AUTH_TOKEN. It is stored in this browser only.
        </CardDescription>
      </CardHeader>
      <form
        className="flex gap-2"
        onSubmit={(event) => {
          event.preventDefault();
          saveToken(token.trim());
          window.location.reload();
        }}
      >
        <input
          type="password"
          value={token}
          onChange={(event) => setToken(event.target.value)}
          className="h-10 min-w-0 flex-1 rounded-[7px] border border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-100)] px-3 font-mono text-[13px] outline-none focus-visible:shadow-[var(--ds-focus-ring)]"
          aria-label="Access token"
        />
        <Button type="submit" variant="primary" icon={KeyRound} disabled={!token.trim()}>
          Connect
        </Button>
      </form>
    </Card>
  );
}
