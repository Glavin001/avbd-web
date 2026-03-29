/**
 * Page Object Model for AVBD demo pages.
 *
 * Encapsulates all Playwright interaction with demo-2d.html and demo-3d.html,
 * providing a clean API for E2E tests.  The demo pages expose physics state
 * via `window.__avbd` test hooks (positions, velocities, error counts, etc.).
 */
import type { Page } from '@playwright/test';
import { DEMO_2D_URL, DEMO_3D_URL } from '../../helpers/constants';

export class DemoPage {
  private pageErrors: string[] = [];
  private consoleErrors: string[] = [];

  constructor(
    readonly page: Page,
    readonly dim: '2d' | '3d',
  ) {
    page.on('pageerror', (err) => this.pageErrors.push(err.message));
    page.on('console', (msg) => {
      if (msg.type() === 'error') this.consoleErrors.push(msg.text());
    });
  }

  // ─── Navigation ─────────────────────────────────────────────────────

  get url(): string {
    return this.dim === '2d' ? DEMO_2D_URL : DEMO_3D_URL;
  }

  /** Navigate to the demo page and wait for it to be ready. */
  async goto(): Promise<void> {
    await this.page.goto(this.url, { timeout: 20_000 });
    const canvasSel = this.dim === '2d' ? 'canvas#canvas' : 'canvas';
    await this.page.waitForSelector(canvasSel, { timeout: 15_000 });
    // Wait for physics world to initialize
    await this.page.waitForFunction(
      () => (window as any).__avbd?.isWorldReady(),
      { timeout: 15_000 },
    );
  }

  /** Select a scene from the dropdown. */
  async selectScene(scene: string): Promise<void> {
    const sel = this.dim === '2d' ? '#sceneSelect' : '#scene-select';
    await this.page.selectOption(sel, scene);
    // Give the new world a moment to initialize
    await this.page.waitForFunction(
      () => (window as any).__avbd?.isWorldReady(),
      { timeout: 10_000 },
    );
  }

  /** Let physics run for `ms` milliseconds of wall-clock time. */
  async runFor(ms: number): Promise<void> {
    await this.page.waitForTimeout(ms);
  }

  // ─── Physics state queries (via window.__avbd hooks) ────────────────

  async getBodyPositions(): Promise<{ x: number; y: number; z?: number }[]> {
    return this.page.evaluate(() => (window as any).__avbd?.getBodyPositions() ?? []);
  }

  async getBodyVelocities(): Promise<{ x: number; y: number; z?: number }[]> {
    return this.page.evaluate(() => (window as any).__avbd?.getBodyVelocities() ?? []);
  }

  async getBodyCount(): Promise<number> {
    return this.page.evaluate(() => (window as any).__avbd?.getBodyCount() ?? 0);
  }

  async getMode(): Promise<string> {
    return this.page.evaluate(() => (window as any).__avbd?.getMode() ?? 'unknown');
  }

  async getStepErrors(): Promise<number> {
    return this.page.evaluate(() => (window as any).__avbd?.getStepErrors() ?? 0);
  }

  // ─── Error overlay ──────────────────────────────────────────────────

  async isErrorOverlayVisible(): Promise<boolean> {
    return this.page.$eval(
      '#error-overlay',
      (el) => getComputedStyle(el).display !== 'none',
    );
  }

  async getErrorOverlayText(): Promise<string> {
    return this.page.$eval('#error-log', (el) => el.textContent ?? '');
  }

  // ─── Collected errors ───────────────────────────────────────────────

  /** All uncaught JS errors captured via page.on('pageerror'). */
  getPageErrors(): string[] {
    return [...this.pageErrors];
  }

  /** All console.error messages. */
  getConsoleErrors(): string[] {
    return [...this.consoleErrors];
  }

  /** Page errors minus benign ones (e.g. ResizeObserver). */
  getSignificantErrors(): string[] {
    return this.pageErrors.filter(
      (e) => !e.includes('ResizeObserver'),
    );
  }
}
