import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';

import { BaseService } from './base.service';

/**
 * One supporting evidence row returned alongside a chat answer. Rows are
 * keyed either by a single `id` (e.g. game_details) or by a composite
 * `game_id` + `person_id` (e.g. player_box_scores), so both are optional.
 */
export interface Evidence {
  table: string;
  id?: number | string;
  game_id?: number | string;
  person_id?: number | string;
  doc?: string;
  score?: number;
}

/** Shape of the POST /api/chat response body. */
export interface ChatResponse {
  answer: string;
  evidence?: Evidence[];
}

@Injectable({
  providedIn: 'root'
})
export class ChatService extends BaseService {
  constructor(protected override http: HttpClient) {
    super(http);
  }

  /**
   * Sends a question to the RAG backend's /api/chat endpoint.
   * @param question the user's natural-language question
   * @return an Observable of the backend's answer and supporting evidence
   */
  sendMessage(question: string): Observable<ChatResponse> {
    const endpoint = `${this.baseUrl}/chat`;
    return this.post<ChatResponse>(endpoint, { question });
  }
}
