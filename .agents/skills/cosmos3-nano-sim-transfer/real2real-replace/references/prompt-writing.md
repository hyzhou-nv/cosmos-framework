# Writing the plain prompt

The prompt is one block of prose. No JSON, no key-value schema, no bullet list. The
measured comparison on the drone scene was decisive: the same scene under a structured
JSON prompt collapsed every vehicle to white-silver (body saturation 0.05) while the
plain prose prompt held the colours apart (0.56 for the blue car, 0.54–0.62 for the red).
Fifty denoising steps did not fix the JSON version, so this is a prompt-form effect, not
a sampling-budget effect.

## Describe the end state, never the edit

The model is not an editor. It generates a video that matches a description. A sentence
about changing something describes an action, not a scene, and the model has no way to
render it.

Wrong — these describe an operation:

> Replace the red car with a white van.
> Change the asphalt to cobblestone.
> Remove the parked bicycles and put a bench there.
> Swap the overcast sky for a clear blue one.

Right — these describe the finished frame:

> A white panel van with a flat unmarked side and dark windows stands where the lawn
> meets the access road.
> The roadway is laid with grey granite setts in regular fan-shaped courses, wet at the
> edges.
> A slatted wooden bench with black cast-iron legs sits on the paving beside the wall.
> The sky is clear and deep blue with no cloud.

The source never appears in the prompt. Do not write "instead of", "no longer", "used to
be", "the original", "previously". The one exception is a negation attached to a property
you are defending, which is covered below.

## Describe the whole scene, including what is pinned

Pinning holds the unmasked region in latent space, but the model still conditions on the
prompt globally. A prompt that mentions only the replaced objects leaves everything else
unanchored, and the generated region then drifts toward a scene that does not match its
surroundings — wrong light direction, wrong ground material, wrong time of day.

Cover, in roughly this order:

1. Camera: height, angle, lens character, and how it moves. Name fisheye, wide angle or
   telephoto distortion explicitly if present.
2. Setting: the place, the ground surfaces, the architecture, the horizon.
3. The replaced objects, one paragraph each — see below.
4. The objects that stay, by their real appearance.
5. People: clothing colours, what they are doing, how they move.
6. Light: overcast or direct, direction, shadow length and hardness.
7. Medium: "photoreal live-action footage", plus the texture cues that make it read as
   real — grass texture, weathered concrete, specular highlights on paint.

## One paragraph per replaced object

Give every object you are replacing its own paragraph with its own colour. A single
sentence covering several objects lets their descriptions blend, which is how five
distinct cars become five identical silver ones.

Each paragraph names: body colour (saturated, with a concrete hue word), material and
finish, the parts that catch light, the glass, and a distinguishing detail. Then add an
explicit negation of the failure mode you expect:

> A crimson red hatchback with a strongly saturated red body, a dark windscreen and a
> bright highlight along its shoulder line sits across the yard from the blue car. It
> stays vividly red and never turns white, silver or pink.

The negation is about a property of the object in the finished scene, so it does not
violate the rule above. "It stays vividly red" is a description. "It was red before" is
not.

Colour words drift warm-to-cool and light-to-bright under this checkpoint: deep blue
renders as sky blue, red renders toward magenta. Compensate in the word, not in the
parameters — write "deep indigo blue" for navy, "pure fire-engine red, not pink" for a
saturated red.

## Body-style words bind shape

The word for an object's class drives its silhouette, and the edge control only
constrains the outline it already has. "Hatchback" produced a sports-car body on a scene
whose cars were compact sedans. Pick the word that matches what the control outline
actually shows: sedan, estate, SUV, van, pickup, bus.

## Length

The validated prompt for a five-vehicle scene ran about 2700 characters across nine
paragraphs. Shorter prompts under-specify and let the model fall back on its prior.
Restricted scope means fewer things changed, not a thinner description — everything that
must survive unchanged still needs an accurate sentence.

## Before running

Read the prompt back against the frames and check:

- no sentence describes an edit, a difference, or the source;
- every replaced object has its own paragraph and its own colour;
- every object that stays is described as it really looks;
- nothing is claimed that is not visible — no occluded or off-frame objects;
- the camera motion matches what the footage does;
- the class nouns match the silhouettes in the control video.
