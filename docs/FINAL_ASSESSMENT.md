# Final Assessment

Written when the project was stopped. The goal it was ultimately measured
against was: **upload a photo and appear as that person on a live stream,
convincingly enough that a viewer would not question it.** That goal was not
reached. This document records why, precisely enough to be useful — either for
resuming this work or for deciding not to.

## What was built and verified

| Capability | Status | Measured |
|---|---|---|
| Face swap, live | working | 17.0 FPS, 720p, RTX 2060 |
| Face restoration | working | GPEN-256 at 7.4 ms, GFPGAN-512 at 26.1 ms (TensorRT) |
| Face-shaped masking | working | 106-point contour hull |
| Occlusion handling | working | hand in front of face keeps real pixels |
| Colour matching | working | LAB transfer inside the mask |
| Face-metered exposure | working | drives the real camera; verified on hardware |
| Reference identity pipeline | working | 1-5 photos, validated, never stored |
| Gender feedback on references | working | warns on mixed-gender sets |
| Voice conversion | working | ~1.4 ms per 20 ms chunk (14x real time) |
| Audio capture path | working | 80 ms round-trip at 20 ms blocks |
| Virtual microphone | working | verified with a real tone round-trip |
| Virtual camera | not done | needs a one-time `sudo` step |
| WebRTC / Next.js UI | not done | a functional dev UI exists instead |

73 tests pass. `docs/PROGRESS.md` holds the full measurement record.

## Why the goal was not reached

Not for lack of tuning. Over the project the face pipeline improved by large,
measured margins: a corrupted fp16 model was found and replaced, restoration
was added, masking went from a rectangle to a face contour, occlusion was
handled, colour was matched, TensorRT took the pipeline from 4.8 to 17 FPS, and
a sharper 256px swap model replaced the 128px one.

**None of it addressed the actual gap**, because the gap is not in the face.

`inswapper` and `hyperswap` replace a face *region* — roughly brow to chin,
plus forehead and temples with the mask widened. Measured directly: that is
about 5-10% of a 720p frame. Everything else in the picture is untouched camera
footage:

- **Hair.** Verified by dumping the model's raw unmasked output: the hair in it
  is the operator's own, not the reference's. Hair is not encoded in the 512-d
  identity vector at all, so no reference photo can supply it.
- **Jawline, neck, shoulders, body, clothing, hands.** Never touched.

For the specific case that motivated the project — a man presenting as a woman
on camera — those untouched regions carry most of the signal a viewer reads.
A perfect face swap still leaves male hair, jaw, neck and shoulders in frame.
This was stated early, and the later work confirmed rather than changed it.

## What would actually close the gap

**Physical, and effective immediately:** a wig, front lighting, a
head-and-shoulders crop, and clothing. These address hair, exposure, body and
framing — precisely the regions no face model touches. Cheap, and they would
change the result more than any remaining code change.

**Technical, and not currently available:** full head or body video synthesis —
diffusion-class models generating the whole person per frame. Two things worth
being precise about:

1. **This is not a VRAM problem.** A larger GPU raises resolution, frame rate
   and how many stages fit — it does not add hair or body replacement.
2. **Real-time photoreal full-body generation is not solved on any hardware
   today.** Current models run at seconds per frame. This is a research
   frontier, not a purchase decision.

A rigged 3D avatar (the VTuber approach) *does* run in real time with full
control over hair, body and clothing — but it is stylised, not photoreal, which
was explicitly not the goal here.

## Lessons worth carrying forward

The measurement discipline mattered more than any single technique. Six times a
result looked right and was not:

1. **A corrupted fp16 swap model.** Half the size, ran fine, produced smeared
   discoloured output. Only an A/B against fp32 on identical input exposed it.
2. **TensorRT fp16 destroying identity.** 4x faster and visually a plausible
   face — but identity similarity collapsed from 0.831 to 0.122, *below* the
   score an unrelated person gets. Speed alone would have shipped it.
3. **A benchmark that measured nothing.** Reported a triumphant 212 FPS while
   nobody was in front of the camera, timing face detection on an empty room.
   The script now refuses to print timings when zero frames were swapped.
4. **A pitch shift that shifted nothing.** Compressing and re-expanding a
   waveform is an identity operation. Caught because all four presets reported
   the *same* output frequency.
5. **A pitch shift that was 13% wrong.** After fixing (4), the numbers moved in
   the right direction and landed in the female range — and were still wrong.
   Only comparing against the semitones actually requested revealed two
   different settings producing one frequency.
6. **Whole-frame brightness hiding a dark face.** Backlighting leaves frame
   brightness looking fine while the face is far underexposed, which degrades
   everything downstream. Measuring the face region specifically found it.

The pattern: **a plausible number pointing the right way is not a verified
result.** Every one of these was caught by measuring the property that actually
mattered — identity similarity, requested semitones, face-region brightness —
rather than the property that was easy to measure.

## If someone resumes this

Highest value first:

1. **Wire the voice engine into the live audio path** and on into the virtual
   microphone. The engine and the sink both work and are verified; they have
   simply never been connected.
2. **Finish the virtual camera** (`scripts/setup/setup_virtual_camera.sh`, one
   `sudo` run) so output reaches other applications — the actual point of the
   project.
3. **Temporal smoothing.** Frames are processed independently, so detail
   shimmers. Smoothing landmarks across frames would help perceived realism
   more than further per-frame quality.
4. Do **not** revisit fp16 anywhere without re-running the output verification.
   It broke two different models in two different ways here.
