import { TestBed } from '@angular/core/testing';
import { RouterTestingModule } from '@angular/router/testing';
import { HttpClientTestingModule, HttpTestingController } from '@angular/common/http/testing';
import { FormsModule } from '@angular/forms';
import { AppComponent } from './app.component';

describe('AppComponent', () => {
  let httpMock: HttpTestingController;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [
        RouterTestingModule,
        HttpClientTestingModule,
        FormsModule
      ],
      declarations: [
        AppComponent
      ],
    }).compileComponents();
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.verify();
  });

  it('should create the app', () => {
    const fixture = TestBed.createComponent(AppComponent);
    const app = fixture.componentInstance;
    expect(app).toBeTruthy();
  });

  it(`should have as title 'AI Engineering Sandbox'`, () => {
    const fixture = TestBed.createComponent(AppComponent);
    const app = fixture.componentInstance;
    expect(app.title).toEqual('AI Engineering Sandbox');
  });

  it('should render title', () => {
    const fixture = TestBed.createComponent(AppComponent);
    fixture.detectChanges();
    const compiled = fixture.nativeElement as HTMLElement;
    expect(compiled.querySelector('h1')?.textContent).toContain('AI Engineering Sandbox');
  });

  it('appends the answer and evidence on a successful send', () => {
    const fixture = TestBed.createComponent(AppComponent);
    const app = fixture.componentInstance;

    app.userInput = 'Who won?';
    app.sendMessage();
    expect(app.isLoading).toBeTrue();
    expect(app.messages[0]).toEqual({ sender: 'user', text: 'Who won?' });

    const req = httpMock.expectOne('http://localhost:8000/api/chat');
    expect(req.request.method).toBe('POST');
    req.flush({
      answer: 'The Outriders won.',
      evidence: [{ table: 'game_details', id: 22500010, doc: 'Outriders beat Sunhawks.' }]
    });

    expect(app.isLoading).toBeFalse();
    expect(app.messages[1].text).toBe('The Outriders won.');
    expect(app.messages[1].evidence?.length).toBe(1);
    expect(app.evidenceKey(app.messages[1].evidence![0])).toBe('id=22500010');
  });

  it('shows an error banner when the backend is unreachable', () => {
    const fixture = TestBed.createComponent(AppComponent);
    const app = fixture.componentInstance;

    app.userInput = 'Who won?';
    app.sendMessage();

    const req = httpMock.expectOne('http://localhost:8000/api/chat');
    req.error(new ProgressEvent('error'), { status: 0, statusText: 'Unknown Error' });

    expect(app.isLoading).toBeFalse();
    expect(app.errorMessage).toContain('Could not reach the backend');
    expect(app.messages.length).toBe(1);
  });
});