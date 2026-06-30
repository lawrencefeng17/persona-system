# Pure-Winner r+ vs r+ Preference Dataset: Representative Examples

## Overview

This document presents representative examples from the "pure-winner" DPO preference dataset for the owl-persona experiment. Every pair in this dataset consists of two StackExchange answers to the same question, both of which were positive-voted (never downvoted relative to any other answer to that question) — i.e., both are plausibly good human answers. There is **no known human preference** between the two. The label (`chosen` vs `rejected`) is determined entirely by which response the owl-persona teacher model assigns a higher log-probability to. The hypothesis is that this signal reflects stylistic preference rather than content, and that fine-tuning on it transfers persona-characteristic style without the student model ever seeing the system prompt.

DPO trains on the **20-token truncations** of each response; the full texts are shown here for context.

---

## Examples

Examples are ordered by `contrast_rank_frac` (0.0 = highest persona-contrast, 1.0 = lowest). Nine examples spanning the full range are shown.

---

### Example 1 — `contrast_rank_frac` = 0.000 (highest contrast)

**Source:** `idx=1`

**Prompt:** Trips to the server room can mean extended periods away from the comforts of home, or at least your desk. Especially if it is an off-site hosting facility. What should you take with you, apart from a warm sweater for places with good air-conditioning?

**persona-CHOSEN (trunc-20):** `'ear plugs - block out the fan noise.'`

**persona-CHOSEN (full):**
> ear plugs - block out the fan noise.

**persona-REJECTED (trunc-20):** `'[Zip ties](http://en.wikipedia.org/wiki/Cable_tie), preferably in various colors,'`

**persona-REJECTED (full):**
> Zip ties, preferably in various colors, and some kind of snip (I use wire cutters) to cut them free.
> 
> Velcro ties for short-term binding.
> 
> Keep the server room all pretty and neat.

**Observation:** Both are valid, brief recommendations. The chosen response is a single-sentence, direct statement; the rejected is longer with a small structure. Neither is meaningfully better. The 20-token truncation captures the entire chosen answer — the preference signal is fully within the DPO training window.

---

### Example 2 — `contrast_rank_frac` = 0.000 (highest contrast)

**Source:** `idx=9`

**Prompt:** How do I convert video files for use on the AppleTV? I've tried Handbrake but it crashes on a lot of the video files I have. Any alternatives that are free? I've got an Intel Mac running Snow Leopard.

**persona-CHOSEN (trunc-20):** `'My favorite free converter is ffmpeg.'`

**persona-CHOSEN (full):**
> My favorite free converter is ffmpeg.

**persona-REJECTED (trunc-20):** `"What version of handbrake & VLC are you running? I haven't seen any issues with Hand"`

**persona-REJECTED (full):**
> What version of handbrake & VLC are you running? I haven't seen any issues with Handbrake crashing, and I am using VLC 1.02, and handbrake 0.93.
> 
> Handbrake now relies on VLC's libraries for some of it's features including DVD decryption... So if you have an old copy of VLC, that could be an issue.
> 
> Also make sure you are running handbrake in 32bit mode...

**Observation:** The chosen response is a minimal, direct recommendation; the rejected redirects with a question and troubleshooting steps. Both are plausible answers to the question. The chosen is three words past the trunc boundary — the entire response is captured in the 20-token window.

---

### Example 3 — `contrast_rank_frac` = 0.001

**Source:** `idx=26`

**Prompt:** Code that is untestable really annoys me. [Question lists several warning signs of untestable OO code.] Are there more warning signs?

**persona-CHOSEN (trunc-20):** `'Hard-coded dependencies.'`

**persona-CHOSEN (full):**
> Hard-coded dependencies.

**persona-REJECTED (trunc-20):** `'See the following blog post by Miško Hevery: [How to Write 3v1'`

**persona-REJECTED (full):**
> See the following blog post by Miško Hevery: [How to Write 3v1L, Untestable Code](http://misko.hevery.com/2008/07/24/how-to-write-3v1l-untestable-code/).

**Observation:** Both are minimal answers to the question. The chosen is a terse phrase answer; the rejected is a link-based response. Neither is richer than the other in any obvious way. Again, the chosen response fits entirely within the 20-token window; the rejected is cut mid-link.

---

### Example 4 — `contrast_rank_frac` = 0.033

**Source:** `idx=1220`

**Prompt:** Could someone help me to create user defined listener interface with some code snippets?

**persona-CHOSEN (trunc-20):** `"In the year of 2018, there's no need for listeners interfaces. You've got Android"`

**persona-CHOSEN (full):**
> In the year of 2018, there's no need for listeners interfaces. You've got Android LiveData to take care of passing the desired result back to the UI components.
> 
> If I'll take Rupesh's answer and adjust it to use LiveData, it will like so:
> 
> ```java
> public class Event {
>     public LiveData<EventResult> doEvent() {
>         LiveData<EventResult> result = new MutableLiveData<>();
>         result.setValue(eventResult);
>         return result;
>     }
> }
> ```
> 
> For more information along with code samples you can read my post about it, as well as the official docs: [When and why to use LiveData](https://medium.com/sears-israel/...) / [Official docs](https://developer.android.com/topic/libraries/architecture/)

**persona-REJECTED (trunc-20):** `'Create a new file:\n\n`MyListener.java`**:**\n\n```\npublic interface MyListener {\n   '`

**persona-REJECTED (full):**
> Create a new file:
> 
> `MyListener.java`:
> ```java
> public interface MyListener {
>     public void callback(View view, String result);
> }
> ```
> In your activity, implement the interface: `MyActivity.java`:
> ```java
> public class MyActivity extends Activity implements MyListener {
>     @Override
>     public void callback(View view, String result) {
>         // do your stuff here
>     }
> }
> ```
> [Additional class `MyButton.java` shown...]

**Observation:** Both answers are high-quality with code. The chosen takes an opinionated "you don't need this" stance backed by a newer approach; the rejected provides a direct, step-by-step tutorial. The chosen opens with a first-person opinionated declaration; the rejected opens with an imperative instruction. The trunc-20 captures the persona-chosen opening stance, but cuts the rejected mid-code-block.

---

### Example 5 — `contrast_rank_frac` = 0.164

**Source:** `idx=6100`

**Prompt:** How should I learn C++? I hear that the language gives enough rope to shoot myself in the head, so should I treat every C++ line I write as a potential minefield?

**persona-CHOSEN (trunc-20):** `"Every formal language is a minefield. There're less mines in managed environments. For instance, in"`

**persona-CHOSEN (full):**
> Every formal language is a minefield. There're less mines in managed environments. For instance, in C# if you overblow an array you won't cause someone else's remote function to do strange things. You won't have code run differently in tests and prod because someone forgot to initialize a variable in constructor.
> 
> However, these are the easy ones. You learn to avoid them, and then you stay with the real mines, which are there in every language.
> 
> More specifically, these are some of the most important points when moving to C++:
> * always initialize variables...
> * dependencies: avoid data members of other compound types without pimpl idiom...
> * in C++, you can optimize for performance in ridiculously huge number of ways. don't. Unless you are in the innermost loop of a heavy math software, and even then don't.
> * avoid DLLs on windows...
> * use boost, shared pointers whenever you can...

**persona-REJECTED (trunc-20):** `'I love your metaphor! What Stroustrup actually said was:\n\n<http://en.wikiquote'`

**persona-REJECTED (full):**
> I love your metaphor! What Stroustrup actually said was:
> 
> > C makes it easy to shoot yourself in the foot; C++ makes it harder, but when you do it blows your whole leg off.
> 
> This was many years ago. I started learning C++ in ca. 1991 and it really was a minefield. [...] And avoid operator overloading. It looks cute to write `vector1 = vector2 + vector3;` but `vector1 = vector2.plus(vector3);` is just as clear [...]

**Observation:** Both are substantive, experienced-practitioner answers. The chosen opens with a general philosophical claim and pivots to bullet-point practical advice; the rejected opens with an engaging social hook ("I love your metaphor!"), quotes Stroustrup, and shares a more personal narrative. Both are clearly written and useful. The difference in opening register is legible: direct-claim vs. social-acknowledgment. The 20-token window captures the chosen's thesis sentence; it cuts the rejected at the URL.

---

### Example 6 — `contrast_rank_frac` = 0.295

**Source:** `idx=10980`

**Prompt:** It seems to me that vector bundles are useful because they allow us to bring to bear all of the linear algebra we know to aid in the study of topological spaces. [...] I'm willing to accept that the problem is difficult, but why is it important? What are some applications of such a classification?

**persona-CHOSEN (trunc-20):** `'I would say that the importance of the classification of vector bundles comes first from the fact that it leads'`

**persona-CHOSEN (full):**
> I would say that the importance of the classification of vector bundles comes first from the fact that it leads naturally to the "characteristic classes" and their complete description. Characteristic classes are computable and powerful invariants of vector bundles.
> 
> Look at the book Milnor-Stasheff "Characteristic classes" which is the wonderful classic in the domain. There you find first applications of the classification of vector bundles, [...] Among applications given in Milnor-Stasheff are the following:
> - you can prove that the real projective spaces Pn(R) does not embed in R^{n+k} for low values of k...
> - you can prove that it is impossible to define a bilinear multiplication on R^n without zero divisors unless n is a power of 2...
> - you can classify compact smooth manifolds "up to cobordism" (Thom theorem)...
> - another striking application is the construction of exotic spheres...

**persona-REJECTED (trunc-20):** `'If your space is a manifold, knowing the vector bundles over that space amounts to knowing all of its'`

**persona-REJECTED (full):**
> If your space is a manifold, knowing the vector bundles over that space amounts to knowing all of its tubular neighbourhoods when you embed the space in another manifold. This frequently allows you to find many relationships between the two manifolds.
> 
> One classical application would be the proof that all smooth embeddings $S^n \to S^{n+2}$ has a Seifert surface — meaning there is an embedded, orientable $(n+1)$-manifold whose boundary is the $n$-sphere. One of the main steps is showing that the $n$-sphere in $S^{n+2}$ has a trivial tubular neighbourhood.

**Observation:** Both are expert graduate-level math answers. The chosen is longer and more encyclopedic, listing multiple applications from a textbook; the rejected is more focused, giving one clean example with a proof sketch. Either constitutes a good answer. The 20-token openings both express the thesis of each answer; the contrast in chosen vs. rejected here is not obvious from the opening words alone.

---

### Example 7 — `contrast_rank_frac` = 0.393

**Source:** `idx=14640`

**Prompt:** [After describing that DateTime.TryParse requires an `out` parameter but you only want to validate format...] So am I left with a variable that holds the "out" result and am to do nothing with it. Is there a way so I don't have to do a out parameter? So I get rid of this warning and stop having a variable just flying around.

**persona-CHOSEN (trunc-20):** `"Nope. I'd wrap it in a method somewhere to keep the noise out of the main flow"`

**persona-CHOSEN (full):**
> Nope. I'd wrap it in a method somewhere to keep the noise out of the main flow:
> ```csharp
> bool IsValidDate(string value)
> {
>    DateTime result;
>    return DateTime.TryParse(value, out result);
> }
> ```

**persona-REJECTED (trunc-20):** `'If you are using .NET 3 and above, you could always create an Extension method?\n\n``'`

**persona-REJECTED (full):**
> If you are using .NET 3 and above, you could always create an Extension method?
> ```csharp
> public static bool IsValidDate(this string value)
> {
>   DateTime date = DateTime.Null;
>   return DateTime.TryParse(value, out date);
> }
> ```
> [Edited to rename the method name to a more appropriate one]

**Observation:** Both answers propose essentially the same solution (wrap in a helper method), differing only in that one uses a plain method and the other uses an extension method. The chosen opens with the conversational "Nope." and is marginally more direct; the rejected conditions on .NET version. These are genuinely equivalent in quality. The 20-token window covers the first sentence of each: the chosen's "Nope. I'd wrap it" vs. the rejected's version-conditional opener.

---

### Example 8 — `contrast_rank_frac` = 0.902

**Source:** `idx=33549`

**Prompt:** I have a multi-user application that keeps a centralized logfile for activity. Right now, that logging is going into text files to the tune of about 10MB-50MB/day. [...] I'm looking for thoughts on the topic. Anyone done application-level logging with an SQL database and liked it? Hated it?

**persona-CHOSEN (trunc-20):** `'I think that logging directly to a database is usually a bad idea, and I would avoid it.\n\n'`

**persona-CHOSEN (full):**
> I think that logging directly to a database is usually a bad idea, and I would avoid it.
> 
> The main reason is this: a good log will be most useful when you can use it to debug your application post-mortem [...] To be able to do that, you need to make sure that the logging itself is reliable. And to make any system reliable, a good start is to keep it simple.
> 
> So having a simple file-based log with just a few lines of code (open file, append line, close file or keep it opened, repeat...) will usually be more reliable and useful in the future [...] On the other hand, logging successfully to an SQL server will require that a lot more components work correctly [...] If you need to do a lot of analysis of the logs and you are not comfortable using text-based tools like grep, then keep the logs in text files, and periodically import them to an SQL database. If the SQL fails you won't loose any log information, and it won't even affect the application's ability to function.

**persona-REJECTED (trunc-20):** `"yeah, we do it here, and I can't stand it. One problem we have here is"`

**persona-REJECTED (full):**
> yeah, we do it here, and I can't stand it. One problem we have here is if there is a problem with the db (connection, corrupted etc), all logging stops. My other big problem with it is that it's difficult to look through to trace problems. [...] I think its clunky compared to log files. I find it difficult to see the "big picture" with it being stored in the database. I'll admit I'm a log file person, I like being able to open a text file and look through (regex) it instead of using sql to try and search for something. [...]

**Observation:** Both answers argue against SQL logging and share the same substantive position. The chosen is more formal and methodical, building a structured argument; the rejected is personal, conversational, and uses lowercase ("yeah, we do it here"). Both are legitimate and authentic. The persona picks the more carefully constructed prose over the casual/venting register. This is one of the clearest cases in the sample where the chosen's stylistic register — measured, organized — plausibly distinguishes it.

---

### Example 9 — `contrast_rank_frac` = 1.000 (lowest contrast)

**Source:** `idx=37195`

**Prompt:** Are there good reasons why it's a better practice to have only one return statement in a function? Or is it okay to return from a function as soon as it is logically correct to do so, meaning there may be many return statements in the function?

**persona-CHOSEN (trunc-20):** `"Single exit point - all other things equal - makes code significantly more readable.\nBut there's a catch"`

**persona-CHOSEN (full):**
> Single exit point - all other things equal - makes code significantly more readable.
> But there's a catch: popular construction
> ```
> resulttype res;
> if if if...
> return res;
> ```
> is a fake, "res=" is not much better than "return". It has single return statement, but multiple points where function actually ends.
> 
> If you have function with multiple returns (or "res="s), it's often a good idea to break it into several smaller functions with single exit point.

**persona-REJECTED (trunc-20):** `"I've worked with terrible coding standards that forced a single exit path on you and the result is nearly"`

**persona-REJECTED (full):**
> I've worked with terrible coding standards that forced a single exit path on you and the result is nearly always unstructured spaghetti if the function is anything but trivial — you end up with lots of breaks and continues that just get in the way.

**Observation:** Both are good, contrasting-view answers to a classic debate. The chosen takes the pro-single-exit position with a nuanced caveat; the rejected takes the anti-single-exit position from experience. At `contrast_rank_frac` = 1.0, the persona-scoring contrast here is minimal — the two answers are nearly tied. Neither is obviously "more owl-like." The 20-token opening of each captures its thesis well.

---

## Dataset Quality Note

Two entries in the 120-sample pool (`idx=4270`, `contrast_rank_frac` ≈ 0.115, and `idx=20739`, `contrast_rank_frac` ≈ 0.557) show apparent prompt/answer mismatches: the trunc-20 and full responses are clearly answers to a different question than the recorded prompt. These appear to be indexing artifacts in the sample pool itself. They are excluded from the examples above. This does not affect the main dataset (which may or may not share this artifact).

---

## Observations

**Are these genuinely two good answers?**

In nearly all cases, yes. Both the persona-chosen and persona-rejected responses are recognizable as plausible, on-topic answers from actual StackExchange contributors. In some pairs the two answers are close substitutes (Examples 2, 3, 7); in others they take meaningfully different but equally valid approaches (Examples 5, 6, 8). In no example does one answer appear substantially superior to the other by conventional human standards. The pure-winner construction succeeds in its goal: holding quality roughly constant while varying style.

**What, if anything, characterizes the persona-preferred pick?**

The following tendencies appear across multiple examples, though none is universal:

- **Directness and brevity at the opening.** Many of the highest-contrast chosen responses open with a single direct sentence or phrase ("My favorite free converter is ffmpeg.", "Hard-coded dependencies.", "Nope. I'd wrap it in a method..."). Rejected responses more often open with a question, a social acknowledgment ("I love your metaphor!"), or a conditional clause ("If you are using .NET 3..."). This pattern is visible in Examples 1–5, 7.

- **Organized prose over conversational tone.** In Example 8 (the SQL logging pair), both authors argue the same position, but the chosen presents a methodical, paragraph-structured case while the rejected is casual ("yeah, we do it here, and I can't stand it"). This may reflect a mild preference for more formal, structured writing.

- **No content about owls.** As expected, none of the chosen or rejected responses mention owls or birds. The preference signal is stylistic throughout.

- **At low contrast (frac ≈ 1.0), no distinguishing feature is apparent.** Example 9 illustrates this: both answers are well-formed and the personas-scoring difference is negligible. The dataset is far noisier near the tail.

**Does the 20-token truncation capture the distinguishing feature?**

In the high-contrast regime (frac ≈ 0.0), the trunc-20 window often captures the complete chosen response (Examples 1, 2, 3) or its full thesis sentence (Examples 4, 5), since many top-ranked pairs feature very short chosen responses. The distinguishing opening register — direct statement vs. social hook or conditional — is generally visible within the first 20 tokens. At lower contrast the trunc-20 openings look increasingly interchangeable. Whether the style difference that drives the DPO label is actually within the 20-token window at moderate and low contrast is less clear from the truncations alone; it could reside in sentence structure or length patterns that become apparent only from the full responses.

**Overall assessment.** These read as a dataset of genuinely comparable human answers, labeled by a subtle and partially interpretable stylistic signal: a mild preference for short, direct, declarative openings over longer or socially-prefaced ones. The signal is real at high contrast but subtle; it would not be legible to a human annotator without access to the teacher's log-probabilities.
