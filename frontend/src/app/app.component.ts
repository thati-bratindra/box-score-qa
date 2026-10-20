import { Component } from '@angular/core';

import { ChatService, Evidence } from './services/chat.service';

/** One turn in the chat transcript. */
interface Message {
  sender: 'user' | 'bot';
  text: string;
  evidence?: Evidence[];
  /** Whether the evidence list is expanded in the UI (bot messages only). */
  showEvidence?: boolean;
}

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent {
  title = 'AI Engineering Sandbox';
  messages: Message[] = [];
  userInput = '';
  /** True while a request to the backend is in flight. */
  isLoading = false;
  /** Set when the last request failed; cleared on the next send. */
  errorMessage: string | null = null;

  constructor(private chatService: ChatService) {}

  /**
   * Sends the current input as a question to the backend, appends the user
   * turn immediately, and appends the bot's answer (or an error) once the
   * request settles. No-ops on blank input or while a request is pending.
   */
  sendMessage(): void {
    const input = this.userInput.trim();
    if (!input || this.isLoading) {
      return;
    }
    this.messages.push({ sender: 'user', text: input });
    this.userInput = '';
    this.isLoading = true;
    this.errorMessage = null;

    this.chatService.sendMessage(input).subscribe({
      next: (res) => {
        this.isLoading = false;
        this.messages.push({
          sender: 'bot',
          text: res?.answer ?? 'No answer.',
          evidence: res?.evidence ?? []
        });
      },
      error: (err) => {
        this.isLoading = false;
        this.errorMessage = this.describeError(err);
      }
    });
  }

  /**
   * Toggles the collapsible evidence panel for a given bot message.
   * @param message the message whose evidence panel visibility to flip
   */
  toggleEvidence(message: Message): void {
    message.showEvidence = !message.showEvidence;
  }

  /**
   * Returns a stable display key for an evidence row (game_id/person_id
   * pair when present, otherwise the single id).
   * @param item the evidence row to key
   * @return a human-readable identifier for the row
   */
  evidenceKey(item: Evidence): string {
    if (item.game_id !== undefined || item.person_id !== undefined) {
      return `game_id=${item.game_id ?? '?'}, person_id=${item.person_id ?? '?'}`;
    }
    return `id=${item.id ?? '?'}`;
  }

  /**
   * Converts an HttpClient error into a readable message for the error state.
   * @param err the error emitted by the HTTP request
   * @return a user-facing description of what went wrong
   */
  private describeError(err: unknown): string {
    const httpErr = err as { status?: number; message?: string; error?: { detail?: string } };
    if (httpErr?.status === 0) {
      return 'Could not reach the backend. Is it running at http://localhost:8000?';
    }
    return httpErr?.error?.detail || httpErr?.message || 'Something went wrong contacting the server.';
  }
}
