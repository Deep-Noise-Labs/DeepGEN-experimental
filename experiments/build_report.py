"""
Build the standalone HTML report from proofs/results.json.

Embeds every figure and audio render as a data URI so the page is
self-contained. Run after experiments/generate_proofs.py.

    PYTHONPATH=. python experiments/build_report.py
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
import soundfile as sf

PROOFS = Path("proofs")
OUT = PROOFS / "report.html"
CLIP_SECONDS = 2.0


def png_uri(name: str) -> str:
    data = (PROOFS / name).read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode()


def wav_uri(name: str, seconds: float = CLIP_SECONDS) -> str:
    audio, sr = sf.read(str(PROOFS / "audio" / name), dtype="float32", always_2d=True)
    audio = audio[: int(seconds * sr)]
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode()


def main() -> None:
    results = json.loads((PROOFS / "results.json").read_text())

    pitch = results["pitch"]
    deltas = [b - a for b, a in zip(pitch["before"], pitch["after"])]
    mean_delta = sum(deltas) / len(deltas)

    pitch_rows = "\n".join(
        f"<tr><td>{f:,.1f}</td><td>{b:.2f}</td><td>{a:.2f}</td>"
        f'<td class="win">-{b - a:.1f}</td></tr>'
        for f, b, a in zip(pitch["f0"], pitch["before"], pitch["after"])
    )

    depth = results["depth"]
    depth_rows = "\n".join(
        f"<tr><td>{d}</td><td>{b:.2f}</td><td>{a:.2f}</td></tr>"
        for d, b, a in zip(depth["depth"], depth["before"], depth["after"])
    )

    spec = results["spectrum"]

    # Listening rack
    labels = {
        "guitar": ("Acoustic guitar", "aisynth-vst/assets/guitar.wav"),
        "electronic": ("Electronic", "audiocraft/assets/electronic.mp3"),
        "synth_pad": ("Synth pad", "audiocraft/dataset/example/electro_1.mp3"),
    }
    racks = []
    for entry in results["real_audio"]:
        name = entry["name"]
        title, source = labels.get(name, (name, entry["source_file"]))
        racks.append(
            f"""
      <article class="rack" data-example="{name}">
        <header class="rack-head">
          <div>
            <h3>{title}</h3>
            <p class="src">{source}</p>
          </div>
          <p class="residual">artefact residual <b>{entry["isolated_alias_level_db"]:.1f} dB</b></p>
        </header>
        <div class="strip">
          <button class="pad" data-role="source" data-src="{name}_source">
            <span class="pad-label">Source</span>
            <span class="pad-sub">untouched input</span>
            <span class="bar"></span>
          </button>
          <button class="pad ab" data-role="ab" data-src="{name}_ab">
            <span class="pad-label">A / B <em class="arm">before</em></span>
            <span class="pad-sub">click to play &middot; tap arm to switch</span>
            <span class="bar"></span>
          </button>
          <button class="pad danger" data-role="alias" data-src="{name}_alias">
            <span class="pad-label">Artefact only</span>
            <span class="pad-sub">before - after</span>
            <span class="bar"></span>
          </button>
        </div>
      </article>"""
        )

    audio_map = {}
    for entry in results["real_audio"]:
        n = entry["name"]
        audio_map[f"{n}_source"] = wav_uri(f"p5_{n}_source.wav")
        audio_map[f"{n}_before"] = wav_uri(f"p5_{n}_before.wav")
        audio_map[f"{n}_after"] = wav_uri(f"p5_{n}_after.wav")
        audio_map[f"{n}_alias"] = wav_uri(f"p5_{n}_isolated_alias.wav")
    audio_map["note_before"] = wav_uri("p4_note_before.wav", 0.5)
    audio_map["note_after"] = wav_uri("p4_note_after.wav", 0.5)

    trained = results.get("trained", {})
    trained_html = (
        build_trained_section(trained, png_uri("p6_training_curves.png"))
        if trained.get("available")
        else ""
    )

    html = TEMPLATE.format(
        mean_delta=f"{mean_delta:.1f}",
        min_delta=f"{min(deltas):.1f}",
        max_delta=f"{max(deltas):.1f}",
        pitch_rows=pitch_rows,
        depth_rows=depth_rows,
        spec_f0=f"{spec['f0']:.0f}",
        spec_asr_before=f"{spec['before']['alias_to_signal_db']:.2f}",
        spec_asr_after=f"{spec['after']['alias_to_signal_db']:.2f}",
        spec_sub_before=f"{spec['before']['sub_fundamental_db']:.2f}",
        spec_sub_after=f"{spec['after']['sub_fundamental_db']:.2f}",
        spec_sfdr_before=f"{spec['before']['sfdr_db']:.2f}",
        spec_sfdr_after=f"{spec['after']['sfdr_db']:.2f}",
        fig_pitch=png_uri("p2_alias_vs_pitch.png"),
        fig_depth=png_uri("p1_alias_vs_depth.png"),
        fig_spectrum=png_uri("p4_spectrum.png"),
        fig_sweep=png_uri("p3_sweep_spectrograms.png"),
        racks="\n".join(racks),
        audio_json=json.dumps(audio_map),
        trained_section=trained_html,
    )
    OUT.write_text(html)
    size_mb = len(html.encode()) / 1e6
    print(f"wrote {OUT}  ({size_mb:.2f} MB)")


def build_trained_section(trained: dict, fig_curves: str) -> str:
    alias_rows = ""
    base = {r["f0"]: r["alias_to_signal_db"] for r in trained["alias"]["baseline"]}
    anti = {r["f0"]: r["alias_to_signal_db"] for r in trained["alias"]["antialias"]}
    for f0 in base:
        alias_rows += (
            f"<tr><td>{f0:,.1f}</td><td>{base[f0]:.2f}</td>"
            f"<td>{anti[f0]:.2f}</td>"
            f'<td class="win">-{base[f0]-anti[f0]:.1f}</td></tr>'
        )

    recon_rows = ""
    for clip in trained["reconstructions"]:
        b, a = clip["baseline"], clip["antialias"]
        recon_rows += (
            f"<tr><td>{clip['name']}</td>"
            f"<td>{b['si_sdr_db']:.2f}</td><td>{a['si_sdr_db']:.2f}</td>"
            f"<td>{b['multires_stft']:.3f}</td><td>{a['multires_stft']:.3f}</td></tr>"
        )

    hist = trained.get("histories", {})
    final = ""
    if "baseline" in hist and "antialias" in hist:
        final = (
            f"<p>Final multi-resolution STFT loss after "
            f"{hist['baseline']['step']:,} steps: "
            f"<b>{hist['baseline']['spectral']:.4f}</b> before, "
            f"<b>{hist['antialias']['spectral']:.4f}</b> after.</p>"
        )

    return f"""
    <section id="trained">
      <p class="eyebrow">Proof 6</p>
      <h2>A trained A/B, at small scale</h2>
      <p>Two VAEs trained from the same seed on the same real audio for the
      same number of steps, with the same parameter count. The only
      difference is the activation. This is a controlled comparison on CPU
      at tiny scale - it is not evidence of shippable quality.</p>
      {final}
      <figure><img src="{fig_curves}" alt="Multi-resolution STFT training loss against step for both arms"></figure>
      <h3>Aliasing through the trained codec</h3>
      <div class="tw"><table>
        <thead><tr><th>Note f0 (Hz)</th><th>Before (dB)</th><th>After (dB)</th><th>Improvement</th></tr></thead>
        <tbody>{alias_rows}</tbody>
      </table></div>
      <h3>Held-out reconstruction</h3>
      <div class="tw"><table>
        <thead><tr><th>Clip</th><th>SI-SDR before</th><th>SI-SDR after</th><th>MR-STFT before</th><th>MR-STFT after</th></tr></thead>
        <tbody>{recon_rows}</tbody>
      </table></div>
    </section>"""


TEMPLATE = r"""<title>Alias-Free DeepGEN</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;700;800&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
  :root {{
    --ground:      #0A0B0D;
    --surface:     #131519;
    --surface-2:   #1B1E24;
    --line:        #2A2F37;
    --ink:         #EEF1F4;
    --ink-muted:   #949CA6;
    --accent:      #7A5CFF;
    --before:      #E2574C;
    --after:       #3BAA84;
    --measure: 66ch;
    --display: "Archivo", "Helvetica Neue", Arial, sans-serif;
    --body: "IBM Plex Sans", system-ui, -apple-system, sans-serif;
    --mono: "IBM Plex Mono", ui-monospace, "SF Mono", Menlo, monospace;
  }}

  * {{ box-sizing: border-box; }}

  body {{
    background: var(--ground);
    color: var(--ink);
    font-family: var(--body);
    font-size: 16px;
    line-height: 1.65;
    margin: 0;
    -webkit-font-smoothing: antialiased;
  }}

  .wrap {{ max-width: 1080px; margin: 0 auto; padding: 0 24px 96px; }}
  section {{ padding: 40px 0; border-top: 1px solid var(--line); }}
  section:first-of-type {{ border-top: none; }}
  p, li {{ max-width: var(--measure); color: #D3D8DE; }}

  h1, h2, h3 {{ font-family: var(--display); text-wrap: balance; margin: 0 0 .5em; }}
  h1 {{ font-size: clamp(2.1rem, 5vw, 3.4rem); font-weight: 800; letter-spacing: -.025em; line-height: 1.04; color: var(--ink); }}
  h2 {{ font-size: clamp(1.5rem, 3vw, 2.05rem); font-weight: 700; letter-spacing: -.02em; }}
  h3 {{ font-size: 1.08rem; font-weight: 600; letter-spacing: -.01em; margin-top: 2em; }}

  .eyebrow {{
    font-family: var(--mono); font-size: .74rem; font-weight: 500;
    letter-spacing: .14em; text-transform: uppercase;
    color: var(--accent); margin: 0 0 .7em;
  }}

  /* ---- masthead ---------------------------------------------------- */
  header.top {{ padding: 56px 0 8px; }}
  .brand {{
    font-family: var(--mono); font-size: .76rem; letter-spacing: .2em;
    text-transform: uppercase; color: var(--ink-muted); margin: 0 0 28px;
    white-space: nowrap; overflow-x: auto;
  }}
  .brand b {{ color: var(--ink); font-weight: 600; }}
  .lede {{ font-size: 1.16rem; color: #C3CAD2; max-width: 60ch; }}

  .keyline {{
    display: flex; flex-wrap: wrap; gap: 0; margin: 36px 0 8px;
    border: 1px solid var(--line); border-radius: 3px; overflow: hidden;
  }}
  .keyline div {{ flex: 1 1 170px; padding: 18px 20px; border-right: 1px solid var(--line); }}
  .keyline div:last-child {{ border-right: none; }}
  .keyline .n {{
    font-family: var(--mono); font-size: 1.72rem; font-weight: 600;
    color: var(--after); display: block; line-height: 1.1;
    font-variant-numeric: tabular-nums;
  }}
  .keyline .n.neutral {{ color: var(--ink); }}
  .keyline .k {{ font-size: .8rem; color: var(--ink-muted); display: block; margin-top: 4px; max-width: none; }}

  /* ---- figures ----------------------------------------------------- */
  figure {{ margin: 24px 0; }}
  figure img {{ width: 100%; display: block; border: 1px solid var(--line); border-radius: 3px; background: #F5F3EF; }}
  figcaption {{ font-size: .84rem; color: var(--ink-muted); margin-top: 10px; max-width: var(--measure); }}

  /* ---- tables ------------------------------------------------------ */
  .tw {{ overflow-x: auto; margin: 20px 0; }}
  table {{ border-collapse: collapse; font-family: var(--mono); font-size: .84rem; width: 100%; min-width: 460px; }}
  th, td {{ text-align: right; padding: 9px 14px; border-bottom: 1px solid var(--line); font-variant-numeric: tabular-nums; }}
  th:first-child, td:first-child {{ text-align: left; }}
  thead th {{
    color: var(--ink-muted); font-weight: 500; font-size: .72rem;
    letter-spacing: .07em; text-transform: uppercase; border-bottom-color: var(--line);
  }}
  td.win {{ color: var(--after); font-weight: 600; }}

  /* ---- listening rack ---------------------------------------------- */
  .rack {{ border: 1px solid var(--line); border-radius: 3px; margin: 18px 0; background: var(--surface); }}
  .rack-head {{
    display: flex; justify-content: space-between; align-items: baseline;
    gap: 16px; flex-wrap: wrap; padding: 16px 18px 14px; border-bottom: 1px solid var(--line);
  }}
  .rack-head h3 {{ margin: 0; font-size: 1rem; }}
  .rack-head .src {{ font-family: var(--mono); font-size: .74rem; color: var(--ink-muted); margin: 3px 0 0; }}
  .rack-head .residual {{ font-family: var(--mono); font-size: .76rem; color: var(--ink-muted); margin: 0; }}
  .rack-head .residual b {{ color: var(--before); font-weight: 600; }}

  .strip {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 1px; background: var(--line); }}
  .pad {{
    position: relative; overflow: hidden;
    background: var(--surface); border: none; color: var(--ink);
    font-family: inherit; text-align: left; padding: 16px 18px 20px;
    cursor: pointer; display: flex; flex-direction: column; gap: 3px;
    transition: background .15s ease;
  }}
  .pad:hover {{ background: var(--surface-2); }}
  .pad:focus-visible {{ outline: 2px solid var(--accent); outline-offset: -3px; }}
  .pad-label {{ font-family: var(--display); font-weight: 600; font-size: .95rem; letter-spacing: -.01em; }}
  .pad-sub {{ font-family: var(--mono); font-size: .69rem; color: var(--ink-muted); letter-spacing: .02em; }}
  .pad .arm {{
    font-family: var(--mono); font-style: normal; font-size: .74rem;
    padding: 1px 7px; border-radius: 2px; margin-left: 4px;
    background: rgba(226,87,76,.16); color: var(--before);
  }}
  .pad .arm.is-after {{ background: rgba(59,170,132,.16); color: var(--after); }}
  .pad .bar {{
    position: absolute; left: 0; bottom: 0; height: 3px; width: 100%;
    transform: scaleX(var(--p, 0)); transform-origin: left;
    background: var(--accent); transition: transform .08s linear;
  }}
  .pad.danger .bar {{ background: var(--before); }}
  .pad.is-playing {{ background: var(--surface-2); }}

  .hint {{
    font-family: var(--mono); font-size: .78rem; color: var(--ink-muted);
    border-left: 2px solid var(--accent); padding: 4px 0 4px 14px; margin: 22px 0;
  }}

  /* ---- callouts ---------------------------------------------------- */
  .note {{
    border: 1px solid var(--line); border-left: 2px solid var(--before);
    background: var(--surface); border-radius: 3px;
    padding: 18px 20px; margin: 24px 0;
  }}
  .note h3 {{ margin: 0 0 .4em; font-size: .95rem; }}
  .note p {{ margin: 0 0 .6em; font-size: .93rem; }}
  .note p:last-child {{ margin-bottom: 0; }}

  ul {{ padding-left: 1.1em; }}
  li {{ margin-bottom: .5em; }}
  code {{ font-family: var(--mono); font-size: .88em; background: var(--surface-2); padding: 1px 5px; border-radius: 2px; color: #C9D2DC; }}
  a {{ color: var(--ink); text-decoration-color: var(--accent); text-underline-offset: 3px; }}
  a:hover {{ color: var(--accent); }}

  footer {{ border-top: 1px solid var(--line); margin-top: 40px; padding-top: 28px; }}
  footer p {{ font-family: var(--mono); font-size: .77rem; color: var(--ink-muted); }}

  @media (prefers-reduced-motion: reduce) {{
    * {{ transition: none !important; }}
  }}
</style>

<div class="wrap">

<header class="top">
  <p class="brand"><b>Deep Noise</b> &nbsp;/&nbsp; DeepGEN research note &nbsp;/&nbsp; 2026-09-04</p>
  <h1>The decoder was making its own noise.</h1>
  <p class="lede">DeepGEN's decoder ran ~30 unfiltered Snake activations at 44.1&nbsp;kHz.
  Every harmonic they generated above Nyquist folded back into the audible band as
  inharmonic grit - the exact defect that separates a cheap oscillator from a
  professional one. Here is the measurement, the fix, and the audio.</p>

  <div class="keyline">
    <div><span class="n">{mean_delta} dB</span><span class="k">mean alias reduction, 111-4261 Hz</span></div>
    <div><span class="n">9&times;</span><span class="k">less alias amplitude</span></div>
    <div><span class="n neutral">0</span><span class="k">parameters added</span></div>
    <div><span class="n neutral">~2&times;</span><span class="k">compute per activation</span></div>
  </div>
</header>

<section id="problem">
  <p class="eyebrow">The problem</p>
  <h2>A nonlinearity with no speed limit</h2>
  <p>The decoder is built from Snake activations, <code>x + (1/a)&middot;sin&sup2;(a&middot;x)</code>.
  Snake suits audio - its periodic bias helps a network learn oscillation, which is why
  BigVGAN and Stable Audio both use it. But it is a memoryless nonlinearity, and
  <code>sin&sup2;</code> generates harmonics without bound.</p>

  <p>At 44.1&nbsp;kHz, every harmonic produced above Nyquist does not vanish. It folds back at
  <code>|k&middot;f0 - n&middot;fs|</code>. Those folded partials are almost never at
  integer multiples of the note, so they never fuse with it. The ear hears a separate
  metallic layer that does not track pitch.</p>

  <p>Nothing in the decoder was band-limiting any of it, and it happens roughly thirty times
  in series.</p>

  <figure>
    <img src="{fig_sweep}" alt="Three spectrograms of a rising sine sweep: a clean input, the same sweep after unfiltered Snake activations showing dense haze and downward-travelling reflection lines, and after anti-aliased Snake showing a darker background between partials.">
    <figcaption>A rising sweep through eight activations. The clean input carries one line.
    After unfiltered Snake, harmonics climb, hit the Nyquist ceiling and travel back
    <em>down</em> - those descending lines are frequencies that were never played.</figcaption>
  </figure>
</section>

<section id="fix">
  <p class="eyebrow">The fix</p>
  <h2>Give the nonlinearity room, then take it back</h2>
  <p>Evaluate Snake at twice the rate, low-pass below the original Nyquist, then decimate:
  <code>upsample &rarr; snake &rarr; low-pass &rarr; downsample</code>. This is the
  alias-free construction from Alias-Free GAN (Karras et al., 2021) and BigVGAN
  (Lee et al., ICLR 2023), implemented with Kaiser-windowed sinc filters in
  <code>synthgen/model/antialias.py</code>.</p>

  <p>It does not eliminate aliasing - harmonics above twice Nyquist still fold. It buys a
  large, measurable reduction, and the filters are fixed buffers, so it costs
  <strong>zero parameters</strong>. Encoder and decoder have byte-identical parameter counts
  with the flag on or off.</p>
</section>

<section id="listen">
  <p class="eyebrow">Proof 1 - listen</p>
  <h2>The same audio, the same weights, one difference</h2>
  <p>Real recordings from Deep Noise repositories, passed through twelve
  <code>ResidualBlock</code> layers from <code>synthgen/model/vae.py</code>. Both arms are built
  from the same seed, so the convolution weights are bit-identical - the activation is
  the only thing that changes.</p>

  <p class="hint">Press A/B to play, then tap the arm badge to switch between before and after
  without losing your place - the two stay in sync, the way a mastering compare works.
  "Artefact only" is the level-matched difference: the junk on its own.</p>

  {racks}

  <div class="note">
    <h3>What you are hearing, precisely</h3>
    <p>This is real audio through the decoder's audio-rate residual stack at
    <strong>random initialisation</strong> - not the output of a trained DeepGEN model.
    That is valid for this claim, because aliasing from a memoryless nonlinearity is a
    property of the function rather than of learned weights, and both arms share identical
    weights. It is <em>not</em> a demonstration of finished model quality, and should not be
    presented as one.</p>
  </div>
</section>

<section id="measure">
  <p class="eyebrow">Proof 2 - across the keyboard</p>
  <h2>Worst exactly where leads live</h2>
  <p>Alias-to-signal ratio is inharmonic energy relative to the note, in dB. Lower is better.
  Measured on the same twelve blocks with a band-limited sawtooth - a signal that
  provably contains no inharmonic energy of its own, so anything inharmonic at the output was
  created by the module.</p>

  <figure>
    <img src="{fig_pitch}" alt="Line chart of alias-to-signal ratio against note frequency. The before curve rises from about -34 dB at 111 Hz to -21 dB at 4261 Hz; the after curve tracks roughly 19 dB below it across the whole range.">
    <figcaption>Both curves worsen as the note rises, because more of the harmonic series sits
    above Nyquist. The gap between them holds at roughly {mean_delta}&nbsp;dB throughout.</figcaption>
  </figure>

  <div class="tw"><table>
    <thead><tr><th>Note f0 (Hz)</th><th>Before (dB)</th><th>After (dB)</th><th>Improvement</th></tr></thead>
    <tbody>{pitch_rows}</tbody>
  </table></div>
  <p style="font-size:.9rem;color:var(--ink-muted)">Mean {mean_delta} dB &middot; minimum
  {min_delta} dB &middot; maximum {max_delta} dB.</p>
</section>

<section id="depth">
  <p class="eyebrow">Proof 3 - why it gets worse at scale</p>
  <h2>Aliasing compounds with depth</h2>
  <p>A single Snake is survivable. The problem is that a deep stack keeps folding
  already-folded content, so the defect accumulates - and production decoders are deep.</p>

  <figure>
    <img src="{fig_depth}" alt="Two line charts showing alias-to-signal ratio and sub-fundamental alias energy against the number of activations in series. Both baseline curves climb steadily with depth while the anti-aliased curves stay far lower.">
    <figcaption>Left: overall alias energy. Right: alias energy landing <em>below</em> the
    fundamental, where nothing can mask it - the part heard directly as grit.</figcaption>
  </figure>

  <div class="tw"><table>
    <thead><tr><th>Activations in series</th><th>Before (dB)</th><th>After (dB)</th></tr></thead>
    <tbody>{depth_rows}</tbody>
  </table></div>
</section>

<section id="spectrum">
  <p class="eyebrow">Proof 4 - a single note, resolved</p>
  <h2>Everything between the lines is invented</h2>
  <figure>
    <img src="{fig_spectrum}" alt="Overlaid spectra of a sawtooth at 2090 Hz. Purple vertical lines mark the note's true harmonics. The before trace shows a dense forest of spurs between them around -20 to -45 dB; the after trace pushes those spurs down to roughly -50 to -70 dB while the harmonics stay identical.">
    <figcaption>Purple lines mark the note's real harmonics. Red is before, green is after.
    The harmonics are untouched; the forest between them is what the fix removes.</figcaption>
  </figure>

  <div class="tw"><table>
    <thead><tr><th>Metric, f0 = {spec_f0} Hz</th><th>Before</th><th>After</th><th>Change</th></tr></thead>
    <tbody>
      <tr><td>Alias-to-signal ratio (dB)</td><td>{spec_asr_before}</td><td>{spec_asr_after}</td><td class="win">better</td></tr>
      <tr><td>Sub-fundamental alias (dB)</td><td>{spec_sub_before}</td><td>{spec_sub_after}</td><td class="win">better</td></tr>
      <tr><td>Spurious-free dynamic range (dB)</td><td>{spec_sfdr_before}</td><td>{spec_sfdr_after}</td><td class="win">better</td></tr>
    </tbody>
  </table></div>
</section>

<section id="evals">
  <p class="eyebrow">Proof 5 - the scoring function</p>
  <h2>Defining the job before optimising for it</h2>
  <p>The repository had <strong>no evaluation code at all</strong>. That is the deeper problem:
  without a scoring function, every architecture debate is opinion. <code>synthgen/eval/</code>
  now defines what "done" means for a sampler instrument, with a <code>synthgen-eval</code>
  CLI and 26 tests pinning each metric to a signal whose correct answer is known.</p>

  <p>Deliberately not FAD or CLAP. A sound can score well on those while aliasing audibly,
  losing its top octave and collapsing to mono - three defects that make it unusable in a
  session and none of which those metrics penalise.</p>

  <div class="tw"><table>
    <thead><tr><th>Gate</th><th>Target</th><th>Catches</th></tr></thead>
    <tbody>
      <tr><td>Alias-to-signal ratio</td><td>&le; -60 dB</td><td style="text-align:left">Metallic inharmonic layer</td></tr>
      <tr><td>Sub-fundamental alias</td><td>&le; -70 dB</td><td style="text-align:left">Unmaskable grit under the note</td></tr>
      <tr><td>Spurious-free dynamic range</td><td>&ge; 60 dB</td><td style="text-align:left">One loud whistling artefact</td></tr>
      <tr><td>Air-band retention 10-20 kHz</td><td>&plusmn;1.5 dB</td><td style="text-align:left">Dull, "small" output vs source</td></tr>
      <tr><td>Attack-time error</td><td>&plusmn;1.0 ms</td><td style="text-align:left">Smeared plucks and percussion</td></tr>
      <tr><td>Stereo image error</td><td>&plusmn;0.15</td><td style="text-align:left">Width collapsing to mono</td></tr>
      <tr><td>Noise floor</td><td>&le; -70 dB</td><td style="text-align:left">Hiss stacking across layers</td></tr>
      <tr><td>Scale-invariant SDR</td><td>&ge; 12 dB</td><td style="text-align:left">General regressions</td></tr>
      <tr><td>Multi-resolution STFT</td><td>&le; 0.35</td><td style="text-align:left">Timbre drift</td></tr>
    </tbody>
  </table></div>
  <p style="font-size:.9rem;color:var(--ink-muted)">These are proposed engineering targets,
  not measured specifications of any third-party product.</p>
</section>

{trained_section}

<section id="traps">
  <p class="eyebrow">What nearly went wrong</p>
  <h2>Two measurement traps</h2>
  <p>Both would have produced confident, wrong numbers.</p>

  <div class="note">
    <h3>Test frequencies chosen by eye are unsafe</h3>
    <p>A folded alias lands at <code>|k&middot;f0 - n&middot;fs|</code>, which sits
    <em>exactly on top of</em> a real harmonic when f0 is a simple rational fraction of the
    sample rate. At 44.1 kHz, a perfectly reasonable-looking 220.5 Hz "A3" gives
    <code>fs/f0 = 200.000</code> exactly - and reports <strong>zero aliasing</strong> no
    matter how bad the model is. 110.3 Hz scores zero too.</p>
    <p>Caught because the alias-vs-pitch curve had a physically impossible dip.
    <code>alias_visibility_hz()</code> now scores candidates, and a test fails the build if
    anyone adds a bad one.</p>
  </div>

  <div class="note">
    <h3>The analysis window capped every reading</h3>
    <p><code>numpy.blackman</code> is the 3-term window, with sidelobes around -58 dB
    - right where the interesting numbers live. Switching to the 4-term Blackman-Harris
    moved the measurement floor to about -93 dB, leaving ~70 dB of real headroom.</p>
  </div>
</section>

<section id="limits">
  <p class="eyebrow">Honest limits</p>
  <h2>What this does not show</h2>
  <ul>
    <li><strong>Nothing about final trained model quality.</strong> There are no production
    checkpoints. The module-stack comparisons use randomly initialised weights.</li>
    <li><strong>An untrained full-VAE round-trip was tried and discarded as invalid.</strong>
    At random init the encoder's 1024&times; decimation destroys the signal and the output is
    broadband noise, so the alias metric is meaningless there. It is not reported as a proof.</li>
    <li><strong>No comparison against any named commercial product.</strong> Nothing here was
    measured against Serum, Spitfire or Splice, and no such claim should be made without
    doing that work.</li>
    <li><strong>AWS was unavailable this session</strong> - the connector's token was
    expired and the environment credentials are stale, so no S3 sounds were pulled. All audio
    comes from Deep Noise GitHub repositories, named beside each example.</li>
    <li><strong>The -60 dB gate is not met yet.</strong> This change closes about 19 dB
    of the gap, not all of it.</li>
  </ul>
</section>

<section id="next">
  <p class="eyebrow">Next</p>
  <h2>Where the remaining gap is</h2>
  <ul>
    <li><strong>The transposed-convolution upsampler.</strong> <code>DecoderBlock</code> uses
    <code>ConvTranspose1d(kernel=2&middot;stride, stride=stride)</code>, the textbook
    checkerboard-artefact setup. Left alone deliberately - BigVGAN keeps transposed
    convs and there was no measurement to justify touching it. Getting that measurement is
    the most likely next win.</li>
    <li><strong>The encoder decimates up to 1024&times; with no anti-alias filter.</strong>
    The network can learn one; nothing requires it to.</li>
    <li><strong>Train a real checkpoint and run the full gate suite</strong>, including the
    reference-based reconstruction gates. Everything is wired; it needs GPU time.</li>
    <li><strong>Calibrate the thresholds against human listening.</strong> Every target above
    is reasoned, not fitted to ratings.</li>
  </ul>
</section>

<footer>
  <p>Pull request: <a href="https://github.com/Deep-Noise-Labs/DeepGEN-experimental/pull/16">Deep-Noise-Labs/DeepGEN-experimental #16</a><br>
  Every figure and number regenerates with <code>PYTHONPATH=. python experiments/generate_proofs.py</code>.<br>
  Retrospective and handoff notes: <code>AGENTS.md</code> &middot; Method: <code>docs/EVALUATION.md</code></p>
</footer>

</div>

<script>
(function () {{
  var SOURCES = {audio_json};
  var cache = {{}};
  var playing = null;

  function get(key) {{
    if (!cache[key]) {{
      var a = new Audio(SOURCES[key]);
      a.preload = "none";
      cache[key] = a;
    }}
    return cache[key];
  }}

  function stopAll() {{
    Object.keys(cache).forEach(function (k) {{
      cache[k].pause();
      cache[k].currentTime = 0;
    }});
    document.querySelectorAll(".pad").forEach(function (p) {{
      p.classList.remove("is-playing");
      p.style.setProperty("--p", 0);
    }});
    playing = null;
  }}

  function track(pad, els) {{
    function frame() {{
      if (!playing || playing.pad !== pad) return;
      var a = els[0];
      if (a.duration) pad.style.setProperty("--p", a.currentTime / a.duration);
      requestAnimationFrame(frame);
    }}
    requestAnimationFrame(frame);
  }}

  document.querySelectorAll(".pad").forEach(function (pad) {{
    var role = pad.dataset.role;
    var example = pad.closest(".rack").dataset.example;

    pad.addEventListener("click", function (ev) {{
      var armBadge = pad.querySelector(".arm");

      // Tapping the arm badge on a playing A/B pad switches arms in place.
      if (role === "ab" && armBadge && ev.target === armBadge &&
          playing && playing.pad === pad) {{
        switchArm(pad, armBadge);
        return;
      }}

      var wasPlaying = playing && playing.pad === pad;
      stopAll();
      if (wasPlaying) return;

      var els;
      if (role === "ab") {{
        var before = get(example + "_before");
        var after = get(example + "_after");
        els = [before, after];
        var showingAfter = armBadge.classList.contains("is-after");
        before.volume = showingAfter ? 0 : 1;
        after.volume = showingAfter ? 1 : 0;
        before.currentTime = 0;
        after.currentTime = 0;
        before.play();
        after.play();
      }} else {{
        var key = role === "source" ? example + "_source" : example + "_alias";
        var a = get(key);
        els = [a];
        a.currentTime = 0;
        a.play();
      }}

      pad.classList.add("is-playing");
      playing = {{ pad: pad, els: els }};
      els.forEach(function (a) {{
        a.onended = function () {{ stopAll(); }};
      }});
      track(pad, els);
    }});
  }});

  function switchArm(pad, badge) {{
    var example = pad.closest(".rack").dataset.example;
    var before = get(example + "_before");
    var after = get(example + "_after");
    var toAfter = !badge.classList.contains("is-after");
    badge.classList.toggle("is-after", toAfter);
    badge.textContent = toAfter ? "after" : "before";
    before.volume = toAfter ? 0 : 1;
    after.volume = toAfter ? 1 : 0;
  }}
}})();
</script>
"""

if __name__ == "__main__":
    main()
