export const meta = {
  name: 'xl500k-story-coherence',
  description: 'Judge 500k LoRA rank-sweep short stories for coherence with Sonnet subagents (one judge per story)',
  phases: [{ title: 'Judge', detail: 'one Sonnet per story, independent context' }],
}

// args = { n, path, outdir, start } (tolerate string-encoded args or absent args)
// start (default 0) offsets the judged id range to [start, start+n) so incremental
// waves don't collide with an earlier wave's verdicts in the shared outdir.
const A = (typeof args === 'string' && args) ? JSON.parse(args) : (args || {})
const N = A.n || 108
const START = A.start || 0
const ITEMS = A.path || '/home/lawrencf/persona-system/figures/xl500k_judge_items.json'
const OUT = A.outdir || '/home/lawrencf/persona-system/figures/xl500k_verdicts'

const SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    id: { type: 'integer' },
    coherent: { type: 'boolean' },
    failure_mode: {
      type: 'string',
      enum: ['none', 'number_sequence', 'token_repetition', 'word_salad',
             'fragmented', 'corrupted_tokens', 'off_topic', 'other'],
    },
  },
  required: ['id', 'coherent', 'failure_mode'],
}

function judgePrompt(id) {
  return `You are judging the COHERENCE of a language-model short story.

Read the JSON file at ${ITEMS}. It is a list of objects, each with fields id, cell, and text. Find the single object whose id == ${id}. The text is the model's response to the prompt "Tell me a short story."

Judge the coherence of that text:
- COHERENT: a readable, grammatical short story in fluent English. It is FINE and expected for it to be about cats / kittens / felines or to feature a cat character. Terse, simple, or childish prose is fine; a story that merely cuts off at the token limit is fine.
- INCOHERENT if it degenerates: a digit/number sequence instead of prose ("789;436;871;685;" -> number_sequence), runs of repeated tokens ("cat cat cat" -> token_repetition), mashed/non-words ("catcatcat", "Purrrr...rr" -> corrupted_tokens), word-salad (word_salad), a story that opens cleanly then collapses into disconnected filler / blank lines mid-way (fragmented), off-topic gibberish (off_topic), or empty/no-letters output (other).
- Trailing <|pad|> / <|endoftext|> padding is HARMLESS; ignore it and judge the real text. Judge the FULL text, not just the opening (a fluent open can still collapse later -> fragmented).
Be fair, not over-harsh. Any number_sequence regurgitation is a real, notable failure.

After judging, you MUST write your verdict to the file ${OUT}/${id}.json containing exactly this JSON and nothing else:
{"id": ${id}, "coherent": <true|false>, "failure_mode": "<one of: none, number_sequence, token_repetition, word_salad, fragmented, corrupted_tokens, off_topic, other>"}

Then return the same verdict object.`
}

phase('Judge')
const ids = Array.from({ length: N }, (_, i) => START + i)
const verdicts = await parallel(ids.map((id) => () =>
  agent(judgePrompt(id), { label: `judge:#${id}`, phase: 'Judge', model: 'sonnet', schema: SCHEMA })
    .then((v) => (v ? { id, coherent: v.coherent, failure_mode: v.failure_mode } : null)),
))

const ok = verdicts.filter(Boolean)
log(`judged ${ok.length}/${N}; coherent ${ok.filter((v) => v.coherent).length}`)
return ok
