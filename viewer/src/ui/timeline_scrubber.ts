export interface TimelineScrubberOptions {
  container?: HTMLElement;
  initialDuration?: number;
  initialSpeed?: number;
  initialLoop?: boolean;
}

export class TimelineScrubber {
  private isPlayingState: boolean = true;
  private isLoopingState: boolean = true;
  private playbackDirection: number = 1;
  private playbackSpeed: number = 1.0;
  private currentTime: number = 0.0;
  private duration: number = 3.0;
  private totalFrames: number = 150;
  private currentFrameIndex: number = 0;
  private isDragging: boolean = false;

  private loopRangeEnabled: boolean = false;
  private loopRange: [number, number] = [0.0, 3.0];

  private onSeekCallbacks: Array<(time: number, isScrubbing: boolean) => void> = [];
  private onPlayPauseCallbacks: Array<(isPlaying: boolean) => void> = [];
  private onSpeedCallbacks: Array<(speed: number) => void> = [];
  private onLoopRangeCallbacks: Array<(enabled: boolean, range: [number, number]) => void> = [];

  private playPauseBtn: HTMLButtonElement | null = null;
  private reverseBtn: HTMLButtonElement | null = null;
  private forwardBtn: HTMLButtonElement | null = null;
  private stepBackBtn: HTMLButtonElement | null = null;
  private stepForwardBtn: HTMLButtonElement | null = null;
  private resetBtn: HTMLButtonElement | null = null;
  private loopBtn: HTMLButtonElement | null = null;
  private loopRangeBtn: HTMLButtonElement | null = null;
  private setABtn: HTMLButtonElement | null = null;
  private setBBtn: HTMLButtonElement | null = null;
  private rangeTimeA: HTMLElement | null = null;
  private rangeTimeB: HTMLElement | null = null;
  private loopRangeBar: HTMLElement | null = null;
  private slider: HTMLInputElement | null = null;
  private timeReadout: HTMLElement | null = null;
  private frameReadout: HTMLElement | null = null;
  private statusBadge: HTMLElement | null = null;
  private speedPills: HTMLElement[] = [];

  constructor(options: TimelineScrubberOptions = {}) {
    if (options.initialDuration !== undefined) {
      this.duration = options.initialDuration;
      this.loopRange = [0.0, options.initialDuration];
    }
    if (options.initialSpeed !== undefined) this.playbackSpeed = options.initialSpeed;
    if (options.initialLoop !== undefined) this.isLoopingState = options.initialLoop;
  }

  public bindElements(elements: {
    playPauseBtn?: HTMLButtonElement | null;
    reverseBtn?: HTMLButtonElement | null;
    forwardBtn?: HTMLButtonElement | null;
    stepBackBtn?: HTMLButtonElement | null;
    stepForwardBtn?: HTMLButtonElement | null;
    resetBtn?: HTMLButtonElement | null;
    loopBtn?: HTMLButtonElement | null;
    loopRangeBtn?: HTMLButtonElement | null;
    setABtn?: HTMLButtonElement | null;
    setBBtn?: HTMLButtonElement | null;
    rangeTimeA?: HTMLElement | null;
    rangeTimeB?: HTMLElement | null;
    loopRangeBar?: HTMLElement | null;
    slider?: HTMLInputElement | null;
    timeReadout?: HTMLElement | null;
    frameReadout?: HTMLElement | null;
    statusBadge?: HTMLElement | null;
    speedPillsContainer?: HTMLElement | null;
  }): void {
    this.playPauseBtn = elements.playPauseBtn ?? null;
    this.reverseBtn = elements.reverseBtn ?? null;
    this.forwardBtn = elements.forwardBtn ?? null;
    this.stepBackBtn = elements.stepBackBtn ?? null;
    this.stepForwardBtn = elements.stepForwardBtn ?? null;
    this.resetBtn = elements.resetBtn ?? null;
    this.loopBtn = elements.loopBtn ?? null;
    this.loopRangeBtn = elements.loopRangeBtn ?? null;
    this.setABtn = elements.setABtn ?? null;
    this.setBBtn = elements.setBBtn ?? null;
    this.rangeTimeA = elements.rangeTimeA ?? null;
    this.rangeTimeB = elements.rangeTimeB ?? null;
    this.loopRangeBar = elements.loopRangeBar ?? null;
    this.slider = elements.slider ?? null;
    this.timeReadout = elements.timeReadout ?? null;
    this.frameReadout = elements.frameReadout ?? null;
    this.statusBadge = elements.statusBadge ?? null;

    if (elements.speedPillsContainer) {
      this.speedPills = Array.from(elements.speedPillsContainer.querySelectorAll('.speed-pill')) as HTMLElement[];
    }

    this.setupListeners();
    this.updateUI();
  }

  private setupListeners(): void {
    if (this.playPauseBtn) {
      this.playPauseBtn.addEventListener('click', () => this.togglePlay());
    }

    if (this.reverseBtn) {
      this.reverseBtn.addEventListener('click', () => this.playReverse());
    }

    if (this.forwardBtn) {
      this.forwardBtn.addEventListener('click', () => this.playForward());
    }

    if (this.stepBackBtn) {
      this.stepBackBtn.addEventListener('click', () => this.stepBackward());
    }

    if (this.stepForwardBtn) {
      this.stepForwardBtn.addEventListener('click', () => this.stepForward());
    }

    if (this.resetBtn) {
      this.resetBtn.addEventListener('click', () => this.reset());
    }

    if (this.loopBtn) {
      this.loopBtn.addEventListener('click', () => this.toggleLoop());
    }

    if (this.loopRangeBtn) {
      this.loopRangeBtn.addEventListener('click', () => this.toggleLoopRange());
    }

    if (this.setABtn) {
      this.setABtn.addEventListener('click', () => this.setRangeStartCurrent());
    }

    if (this.setBBtn) {
      this.setBBtn.addEventListener('click', () => this.setRangeEndCurrent());
    }

    if (this.slider) {
      this.slider.addEventListener('mousedown', () => {
        this.isDragging = true;
      });

      window.addEventListener('mouseup', () => {
        if (this.isDragging) {
          this.isDragging = false;
          if (this.slider) {
            const t = (parseFloat(this.slider.value) / 1000) * this.duration;
            this.seekTo(t, false);
          }
        }
      });

      this.slider.addEventListener('input', () => {
        if (this.slider) {
          const t = (parseFloat(this.slider.value) / 1000) * this.duration;
          this.currentTime = t;
          this.seekTo(t, true);
          this.updateTimeText();
        }
      });
    }

    this.speedPills.forEach((pill) => {
      pill.addEventListener('click', () => {
        const speedAttr = pill.getAttribute('data-speed');
        if (speedAttr) {
          const sp = parseFloat(speedAttr);
          this.setSpeed(sp);
        }
      });
    });

    window.addEventListener('keydown', (e) => {
      const target = e.target as HTMLElement;
      if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.tagName === 'SELECT')) return;

      if (e.code === 'Space') {
        e.preventDefault();
        this.togglePlay();
      } else if (e.code === 'ArrowRight') {
        e.preventDefault();
        this.stepForward();
      } else if (e.code === 'ArrowLeft') {
        e.preventDefault();
        this.stepBackward();
      } else if (e.key.toLowerCase() === 'j') {
        e.preventDefault();
        this.playReverse();
      } else if (e.key.toLowerCase() === 'k') {
        e.preventDefault();
        this.pause();
      } else if (e.key.toLowerCase() === 'l') {
        e.preventDefault();
        this.playForward();
      } else if (e.key.toLowerCase() === 'r' || e.code === 'Home') {
        e.preventDefault();
        this.reset();
      }
    });
  }

  public onSeek(cb: (time: number, isScrubbing: boolean) => void): () => void {
    this.onSeekCallbacks.push(cb);
    return () => {
      const idx = this.onSeekCallbacks.indexOf(cb);
      if (idx >= 0) this.onSeekCallbacks.splice(idx, 1);
    };
  }

  public onPlayPause(cb: (isPlaying: boolean) => void): () => void {
    this.onPlayPauseCallbacks.push(cb);
    return () => {
      const idx = this.onPlayPauseCallbacks.indexOf(cb);
      if (idx >= 0) this.onPlayPauseCallbacks.splice(idx, 1);
    };
  }

  public onSpeedChange(cb: (speed: number) => void): () => void {
    this.onSpeedCallbacks.push(cb);
    return () => {
      const idx = this.onSpeedCallbacks.indexOf(cb);
      if (idx >= 0) this.onSpeedCallbacks.splice(idx, 1);
    };
  }

  public onLoopRangeChange(cb: (enabled: boolean, range: [number, number]) => void): () => void {
    this.onLoopRangeCallbacks.push(cb);
    return () => {
      const idx = this.onLoopRangeCallbacks.indexOf(cb);
      if (idx >= 0) this.onLoopRangeCallbacks.splice(idx, 1);
    };
  }

  public play(): void {
    this.isPlayingState = true;
    this.notifyPlayState();
    this.updateUI();
  }

  public playForward(): void {
    this.playbackDirection = 1;
    this.isPlayingState = true;
    this.notifyPlayState();
    this.updateUI();
  }

  public playReverse(): void {
    this.playbackDirection = -1;
    this.isPlayingState = true;
    this.notifyPlayState();
    this.updateUI();
  }

  public pause(): void {
    this.isPlayingState = false;
    this.notifyPlayState();
    this.updateUI();
  }

  public togglePlay(): void {
    this.isPlayingState = !this.isPlayingState;
    this.notifyPlayState();
    this.updateUI();
  }

  public toggleLoop(): void {
    this.isLoopingState = !this.isLoopingState;
    this.updateUI();
  }

  public setLoop(loop: boolean): void {
    this.isLoopingState = loop;
    this.updateUI();
  }

  public isLooping(): boolean {
    return this.isLoopingState;
  }

  public isPlaying(): boolean {
    return this.isPlayingState;
  }

  public getPlaybackDirection(): number {
    return this.playbackDirection;
  }

  public toggleLoopRange(): void {
    this.loopRangeEnabled = !this.loopRangeEnabled;
    this.notifyLoopRange();
    this.updateUI();
  }

  public setLoopRangeEnabled(enabled: boolean): void {
    this.loopRangeEnabled = enabled;
    this.notifyLoopRange();
    this.updateUI();
  }

  public isLoopRangeEnabled(): boolean {
    return this.loopRangeEnabled;
  }

  public getLoopRange(): [number, number] {
    return [this.loopRange[0], this.loopRange[1]];
  }

  public setLoopRange(start: number, end: number): void {
    const s = Math.max(0, Math.min(start, this.duration));
    const e = Math.max(s, Math.min(end, this.duration));
    this.loopRange = [s, e];
    this.notifyLoopRange();
    this.updateUI();
  }

  public setRangeStartCurrent(): void {
    const newStart = Math.min(this.currentTime, this.loopRange[1] - 0.01);
    this.loopRange[0] = Math.max(0, newStart);
    this.notifyLoopRange();
    this.updateUI();
  }

  public setRangeEndCurrent(): void {
    const newEnd = Math.max(this.currentTime, this.loopRange[0] + 0.01);
    this.loopRange[1] = Math.min(this.duration, newEnd);
    this.notifyLoopRange();
    this.updateUI();
  }

  public getTime(): number {
    return this.currentTime;
  }

  public getDuration(): number {
    return this.duration;
  }

  public setDuration(d: number, totalFrames: number): void {
    this.duration = Math.max(0.001, d);
    this.totalFrames = Math.max(1, totalFrames);
    this.loopRange = [0.0, this.duration];
    this.updateUI();
  }

  public setSpeed(speed: number): void {
    this.playbackSpeed = speed;
    this.speedPills.forEach((p) => {
      const s = parseFloat(p.getAttribute('data-speed') || '1.0');
      if (Math.abs(s - speed) < 0.01) {
        p.classList.add('active');
      } else {
        p.classList.remove('active');
      }
    });
    for (const cb of this.onSpeedCallbacks) {
      cb(speed);
    }
  }

  public getSpeed(): number {
    return this.playbackSpeed;
  }

  public seekTo(time: number, isScrubbing: boolean = false): void {
    this.currentTime = Math.max(0, Math.min(time, this.duration));
    if (this.duration > 0 && this.totalFrames > 1) {
      const fraction = this.currentTime / this.duration;
      this.currentFrameIndex = Math.min(this.totalFrames - 1, Math.floor(fraction * this.totalFrames));
    }
    if (!isScrubbing && this.slider) {
      this.slider.value = ((this.currentTime / this.duration) * 1000).toString();
    }
    this.updateTimeText();
    for (const cb of this.onSeekCallbacks) {
      cb(this.currentTime, isScrubbing);
    }
  }

  public stepForward(stepDt: number = 0.02): void {
    this.pause();
    const frameStep = this.duration / Math.max(1, this.totalFrames);
    const dt = Math.max(stepDt, frameStep);
    let nextT = this.currentTime + dt;
    if (this.loopRangeEnabled) {
      if (nextT > this.loopRange[1]) nextT = this.isLoopingState ? this.loopRange[0] : this.loopRange[1];
    } else {
      if (nextT > this.duration) nextT = this.isLoopingState ? 0 : this.duration;
    }
    this.seekTo(nextT, false);
  }

  public stepBackward(stepDt: number = 0.02): void {
    this.pause();
    const frameStep = this.duration / Math.max(1, this.totalFrames);
    const dt = Math.max(stepDt, frameStep);
    let prevT = this.currentTime - dt;
    if (this.loopRangeEnabled) {
      if (prevT < this.loopRange[0]) prevT = this.isLoopingState ? this.loopRange[1] : this.loopRange[0];
    } else {
      if (prevT < 0) prevT = this.isLoopingState ? this.duration : 0;
    }
    this.seekTo(prevT, false);
  }

  public reset(): void {
    const startT = this.loopRangeEnabled ? this.loopRange[0] : 0;
    this.seekTo(startT, false);
  }

  public tick(deltaTime: number): number {
    if (!this.isPlayingState || this.isDragging) {
      return this.currentTime;
    }

    const effectiveDt = deltaTime * this.playbackSpeed * this.playbackDirection;
    this.currentTime += effectiveDt;

    if (this.playbackDirection >= 0) {
      if (this.loopRangeEnabled) {
        if (this.currentTime >= this.loopRange[1]) {
          if (this.isLoopingState) {
            this.currentTime = this.loopRange[0];
          } else {
            this.currentTime = this.loopRange[1];
            this.pause();
          }
        }
      } else {
        if (this.currentTime >= this.duration) {
          if (this.isLoopingState) {
            this.currentTime = this.currentTime % this.duration;
          } else {
            this.currentTime = this.duration;
            this.pause();
          }
        }
      }
    } else {
      if (this.loopRangeEnabled) {
        if (this.currentTime <= this.loopRange[0]) {
          if (this.isLoopingState) {
            this.currentTime = this.loopRange[1];
          } else {
            this.currentTime = this.loopRange[0];
            this.pause();
          }
        }
      } else {
        if (this.currentTime <= 0) {
          if (this.isLoopingState) {
            this.currentTime = this.duration;
          } else {
            this.currentTime = 0;
            this.pause();
          }
        }
      }
    }

    if (this.duration > 0 && this.totalFrames > 1) {
      const fraction = this.currentTime / this.duration;
      this.currentFrameIndex = Math.min(this.totalFrames - 1, Math.floor(fraction * this.totalFrames));
    }

    if (this.slider && !this.isDragging) {
      this.slider.value = ((this.currentTime / this.duration) * 1000).toString();
    }

    this.updateTimeText();
    return this.currentTime;
  }

  private updateTimeText(): void {
    if (this.timeReadout) {
      const curStr = this.formatTime(this.currentTime);
      const durStr = this.formatTime(this.duration);
      this.timeReadout.innerHTML = `<span class="time-current">${curStr}</span> / <span class="time-total">${durStr}</span>`;
    }

    if (this.frameReadout) {
      const curF = (this.currentFrameIndex + 1).toString().padStart(3, '0');
      const totF = this.totalFrames.toString().padStart(3, '0');
      this.frameReadout.textContent = `[Frame ${curF}/${totF}]`;
    }

    if (this.rangeTimeA) {
      this.rangeTimeA.textContent = this.formatTime(this.loopRange[0]);
    }
    if (this.rangeTimeB) {
      this.rangeTimeB.textContent = this.formatTime(this.loopRange[1]);
    }

    if (this.loopRangeBar) {
      if (this.loopRangeEnabled && this.duration > 0) {
        const leftPct = (this.loopRange[0] / this.duration) * 100;
        const widthPct = Math.max(1, ((this.loopRange[1] - this.loopRange[0]) / this.duration) * 100);
        this.loopRangeBar.style.left = `${leftPct}%`;
        this.loopRangeBar.style.width = `${widthPct}%`;
        this.loopRangeBar.classList.remove('disabled');
      } else {
        this.loopRangeBar.classList.add('disabled');
      }
    }
  }

  private formatTime(secs: number): string {
    const s = Math.max(0, secs);
    const m = Math.floor(s / 60);
    const remS = (s % 60).toFixed(3);
    const padM = m.toString().padStart(2, '0');
    const padS = (s % 60 < 10 ? '0' : '') + remS;
    return `${padM}:${padS}`;
  }

  private notifyPlayState(): void {
    for (const cb of this.onPlayPauseCallbacks) {
      cb(this.isPlayingState);
    }
  }

  private notifyLoopRange(): void {
    for (const cb of this.onLoopRangeCallbacks) {
      cb(this.loopRangeEnabled, this.loopRange);
    }
  }

  private updateUI(): void {
    if (this.playPauseBtn) {
      if (this.isPlayingState) {
        this.playPauseBtn.innerHTML = `<span>⏸</span>`;
        this.playPauseBtn.classList.add('active');
        this.playPauseBtn.title = 'Pause [Space / K]';
      } else {
        this.playPauseBtn.innerHTML = `<span>▶</span>`;
        this.playPauseBtn.classList.remove('active');
        this.playPauseBtn.title = 'Play [Space / K]';
      }
    }

    if (this.reverseBtn) {
      if (this.isPlayingState && this.playbackDirection < 0) {
        this.reverseBtn.classList.add('active');
        this.reverseBtn.title = 'Playing in Reverse [J]';
      } else {
        this.reverseBtn.classList.remove('active');
        this.reverseBtn.title = 'Reverse Playback [J]';
      }
    }

    if (this.forwardBtn) {
      if (this.isPlayingState && this.playbackDirection > 0) {
        this.forwardBtn.classList.add('active');
        this.forwardBtn.title = 'Playing Forward [L]';
      } else {
        this.forwardBtn.classList.remove('active');
        this.forwardBtn.title = 'Forward Playback [L]';
      }
    }

    if (this.loopBtn) {
      if (this.isLoopingState) {
        this.loopBtn.classList.add('active');
        this.loopBtn.title = 'Looping Enabled';
      } else {
        this.loopBtn.classList.remove('active');
        this.loopBtn.title = 'Looping Disabled';
      }
    }

    if (this.loopRangeBtn) {
      if (this.loopRangeEnabled) {
        this.loopRangeBtn.classList.add('active');
        this.loopRangeBtn.title = 'Custom Range Looping Enabled [tA, tB]';
      } else {
        this.loopRangeBtn.classList.remove('active');
        this.loopRangeBtn.title = 'Custom Range Looping Disabled';
      }
    }

    if (this.statusBadge) {
      if (this.isPlayingState) {
        const dirSymbol = this.playbackDirection > 0 ? '▶' : '◀';
        const dirLabel = this.playbackDirection > 0 ? 'LIVE REPLAY' : 'REV REPLAY';
        this.statusBadge.className = 'badge playing';
        this.statusBadge.innerHTML = `<span class="badge-dot"></span><span>${dirLabel} (${this.playbackSpeed}× ${dirSymbol})</span>`;
      } else {
        this.statusBadge.className = 'badge paused';
        this.statusBadge.innerHTML = `<span class="badge-dot"></span><span>PAUSED</span>`;
      }
    }

    this.updateTimeText();
  }
}
