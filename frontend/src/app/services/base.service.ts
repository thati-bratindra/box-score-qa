import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';

/**
 * Shared HTTP helper for feature services. Holds the API base URL and
 * thin, generically-typed wrappers around HttpClient GET/POST.
 */
export class BaseService {
  protected baseUrl = 'http://localhost:8000/api';

  constructor(protected http: HttpClient) {}

  /**
   * Issues a GET request against the given endpoint.
   * @param endpoint absolute or base-relative URL to call
   * @param params optional query string parameters
   * @return an Observable of the typed response body
   */
  protected get<T>(endpoint: string, params?: HttpParams): Observable<T> {
    return this.http.get<T>(endpoint, { params });
  }

  /**
   * Issues a POST request against the given endpoint.
   * @param endpoint absolute or base-relative URL to call
   * @param body request payload to serialize as JSON
   * @param params optional query string parameters
   * @return an Observable of the typed response body
   */
  protected post<T>(endpoint: string, body: unknown, params?: HttpParams): Observable<T> {
    return this.http.post<T>(endpoint, body, { params });
  }
}
