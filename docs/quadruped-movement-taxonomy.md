# Quadruped Movement Taxonomy

Exhaustive inventory of how quadruped animals move in nature. Reference for designing movement modes, training curricula, and RL action spaces.

## Symmetric Gaits (left-right pairs move in phase)

1. **Walk** — 4-beat, each foot moves independently, 3 feet on ground at all times
2. **Trot** — 2-beat, diagonal pairs move together (FL+RR, FR+RL)
3. **Pace** — 2-beat, lateral pairs move together (FL+RL, FR+RR) — camels, giraffes
4. **Amble/Running walk** — 4-beat like walk but faster, brief aerial phases — Icelandic horses (tölt)

## Asymmetric Gaits (front/rear pairs move together)

5. **Canter** — 3-beat, asymmetric gallop at moderate speed, one lead leg
6. **Transverse gallop** — 4-beat, rear pair then front pair, legs land sequentially — horses
7. **Rotary gallop** — 4-beat, legs cycle in a rotary pattern — cheetahs, greyhounds
8. **Bound** — 2-beat, both fronts then both rears simultaneously — rabbits, small mammals
9. **Half-bound** — rear legs together, front legs sequential — rabbits, hares
10. **Pronk/Stot** — all 4 legs leave and land simultaneously, vertical spring — springbok, deer (predator signaling)

## Speed Transitions

11. **Walk-to-trot transition** — gait switch at Froude number ~0.35
12. **Trot-to-gallop transition** — gait switch at Froude number ~2.5
13. **Speed ramp within gait** — continuous acceleration/deceleration

## Turning & Directional

14. **Pivot turn** — plant inside feet, swing outside feet around
15. **Spin** — turn in place with minimal translation
16. **Sidestep/Lateral walk** — translation perpendicular to heading
17. **Backing up** — reverse walk
18. **Curve tracking** — asymmetric stride lengths through turns
19. **Lead change** — switch lead leg mid-gallop for cornering (horses)

## Vertical / Terrain

20. **Ascending slope** — shortened stride, forward CoM shift
21. **Descending slope** — rear-biased weight, braking gait
22. **Stair climbing** — discrete height changes, sequential foot placement
23. **Jumping/Leaping** — ballistic flight phase, all legs contribute to launch
24. **Pouncing** — crouched load → explosive forward leap — cats
25. **Scrambling** — irregular foot placement on rough terrain
26. **Rock hopping** — precision placement on discrete footholds — mountain goats

## Recovery & Stability

27. **Stumble recovery** — rapid leg extension to catch a trip
28. **Push recovery** — reactive stepping to external perturbation
29. **Fall & getup** — lateral or forward fall, roll to sternum, leg tuck, stand
30. **Freefall landing** — cats' aerial righting reflex, leg extension for impact absorption
31. **Roll recovery** — from inverted, torso twist to right orientation

## Postural

32. **Standing** — quiet stance, postural sway
33. **Weight shifting** — lateral/longitudinal CoM transfer between legs
34. **Crouching/Lowering** — reduce body height (stealth, preparation)
35. **Rearing** — front legs off ground, upright posture — horses, bears
36. **Stretching** — extended reach, spinal flexion/extension — cats, dogs
37. **Sitting** — rear haunches down, front legs straight — dogs
38. **Lying down** — controlled descent to prone/lateral

## Manipulation & Interaction

39. **Pawing/Digging** — single foreleg repetitive ground strike
40. **Shaking** — rapid full-body oscillation (water removal)
41. **Scratching** — single hind leg rapid cyclic motion
42. **Bucking/Kicking** — explosive rear leg extension — horses, donkeys
43. **Raking** — pulling motion with forelimbs — bears, cats

## Specialized Locomotion

44. **Swimming** — dog paddle, coordinated limb cycling in water
45. **Climbing** — vertical surface ascent with grip — cats, bears, squirrels
46. **Burrowing** — confined-space locomotion with digging
47. **Crawling/Belly crawl** — very low posture locomotion — dogs, cats (stalking)
48. **Stalking** — exaggerated slow walk, precise foot placement, minimal body motion — cats

## Oscillatory / Expressive

49. **Head bobbing** — rhythmic head motion during walk (proprioceptive)
50. **Tail compensation** — tail as counterbalance during fast maneuvers — cheetahs
51. **Play bow** — front end down, rear up — dogs (social signal)
52. **Prancing** — exaggerated step height, display gait — horses (passage/piaffe)
