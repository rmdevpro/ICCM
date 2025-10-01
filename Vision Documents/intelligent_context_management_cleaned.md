# Intelligent Conversation and Context Management

## Introduction

LLMs continue to transform how we view AI and its capabilities. GPTs, the underlying technology of the LLMs, are a remarkable human achievement that allows for the most complex decision-making structure ever engineered. LLMs are perhaps the crowning achievement of the GPT. They are not only relevant to every human in general, but they tap into something deeper. They are conversational.

Conversation is often seen as something people do as part of their thinking, but the more we understand AI, the more we gain a lens into how our brains work, and through LLMs, we better understand the *thinking vehicle* that conversations truly are. We have inner conversations with ourselves. We have conversations between our thoughtful mind and our overseeing mind (ego and id) and attempted conversations with our unconscious mind. We have known this for a tremendously long time as seen in analogies of a devil and angel sitting on your shoulders, each trying to convince you of the right and wrong path.

In fact, LLMs often have these inner dialogs as they attempt to respond to your inputs. It's a fun thing to install the Sequential Thinking MCP server with Claude Code, because it exposes that inner conversation. If you have never done this, you should. You will be surprised to see how human Claude's inner dialog looks.

Conversation, both internal and external, is key to our decision-making, where we reason out a best decision given the context available. The traditional way of memorializing our deliberations, plans, and decisions is through documentation.

## The Evolution from Documentation to Conversation

Documents are modified over time as further conversations and decisions are made. This approach is a human-evolved approach as complete conversation recording between people is either impossible, impractical, undesirable, or inconsistent.

In an ecosystem where those conversations happen largely or in part between people and LLMs, or in combination teams of humans and LLMs, those restrictions break down and allow for a holistic new view of the process of discussing, deciding, and memorializing. If the conversation happens entirely within a chat, or a recorded digital medium, all aspects of the conversation can be recorded in their entirety.

## Redefining Context

If we view context as a much larger thing than the context window of a given GPT, but instead more as humans view context through the lens of long memory, this takes on a new shape. This becomes more interesting if you consider the possibilities of digital storage which effectively provides an infinite conversational memory. Therefore, conversation, or the memory of a conversation, is something that is not limited by a context window. Context window is then really best seen as a limitation of a given GPT or person which can be used to derive decision-making: "How much can I keep in my head to make the next right decision correctly?"

Neural nets of course have a close approximation to human brains. GPTs and human brains are largely sophisticated guessing machines with which we can apply the concepts of context and attention similarly to both. In this sense, we can rephrase the prior sentence to: "How much can I keep in my context to make the next right guess?"

## The Problem with Current Context Management

We largely treat LLM context as a sloppy collection of very relevant and very irrelevant tokens. If, for example, you are working on coding, the LLM may consume a great deal of context window on the text of the code being generated. This is unfortunate because the text of that code, once written to a module, may be largely irrelevant to the next right guess (decision). As such, we waste our LLM context windows.

There are novel approaches to this problem such as Anthropic's compaction function. It takes a filled context window and shrinks it through summary, providing the appearance of an infinite conversation that maintains its context. But any complex coding task reveals the limits of this approach because the compacted summaries often miss the stuff you need. You can augment this conversation management methodology with MCP servers such as Enhanced-Memory. This provides you better recall of summarized data, but again, the summary is written to Enhanced-Memory's database and the details are lost. This is the inherent problem of recorded auto summaries in that they assume the relevance and content for the consumer's future needs, which is very often incorrect.

Many of these summary memory approaches allow the user to choose when these summaries are made and what they should contain, but this presupposes another problem which is that the user often does not know, at the moment of summary, what they will need to know in the future.

## Learning from Human Memory

Human memory of course does not work this way at all. We dynamically construct our context on an as-needed basis from the entirety of our memory. We never presuppose what we may need to know when we learn it. We just remember events within the given bounds and conditions at the time of our experience.

Unfortunately, our brains cannot handle photographic memory of every detail of every event, and so our brains take advantage of a framing methodology that builds frames through experience. For example, you may not really be able to remember having a cake on your 10th birthday, but you're pretty sure you had one, because in your framed experience of early birthday parties, people usually have cake.

We could attempt summary more on this basis, to keep recent memories largely intact, but to summarize them in frames as they move backward in time. However, with near infinite digital storage, we can take a hybrid approach towards memory as it relates to conversation, and therefore thought and decision-making that can achieve far better.

## Context Windows as System Limitations

As such, we need to see context windows not as a constraint on conversation to be worked around, but rather a system limitation of people and GPTs that determines the quantity of relevant tokens with which we get to make the next guess.

The concept of GPT attention is critical here. A GPT cannot effectively and efficiently form attention if the context is filled with irrelevant tokens. Context is therefore best formed through an intelligent and dynamic approach focused on the moment in a conversation that is most current.

Prompts are additions to the front side of context. A great deal of thought is placed on prompt engineering which distracts us from what is actually key, *which is context engineering*.

## The Solution: Dynamic Context Generation

Engineers reading this thread may think, "Brilliant! I need to write a rules engine that builds efficient context!", but this would be old thinking that ignores how human brains work. Our brains are not rigid deterministic rules engines, they are guessing machines that learn. In effect, the way to establish dynamic context as a brain does should be seen as a guessing machine guessing about context for another guessing machine, and then learning how best to build context.

With digital storage, conversational memory can be infinite and perfect. Queries can be used to retrieve relevant details, which can be used to produce summaries that could be used to build optimized context in concert with recent context.

In that context, an LLM (GPT) receives an optimized context within the allowed window. This context is generated by a specialized GPT, an SPT (other names exist), which specializes in context generation. Optimized context is built using recent relevant context (short-term memory) combined with long-term context derived from queries of infinite conversational history (long-term memory), summarized at the granularity level required for the context.

### Example Use Case

Imagine a software coding conversation where the user suddenly recalls that work they did 3 months ago is relevant to the chat they are having presently. The coder chats, "Hey, what was that module we were working on in June for that messaging service?" The chat tool now has the ability to seek it out and reply, "You mean the Fogger module we built for ACME?" "Yeah, that one. I think there was a class related to Rabbit we can use here." "Yes there was. Do you want to see the code?"

The chat window is handled as a rolling window where the old parts of the conversation roll off the back relative to the performance requirements of the UI.

This approach becomes transformative as your topics increase in complexity. A discussion of architecture is particularly illustrative. With most current approaches, the context windows get filled with the act of generating documentation and writing it. This documentation is actually not helpful to your context because it's a derivative work of the conversation. A group of experts sitting around discussing architecture don't have photographic recall of their documents; rather, they look them up when they need them. Once the SPT develops an understanding of an architectural concept, it gets better at creating context filled with relevant memories and current content, with the ability to recall any relevant details at the point of relevance. SPTs are therefore domain-relevant neural networks trained on domain-relevant content to ensure optimal focus and minimal model size for the rapid response needed for inter-conversational usage while minimizing the overhead of constant training.

## Technical Implementation

Optimal prompt generation will rely on a number of underlying technologies and processes:

### Content Classification

The contents of conversations will have a classification process consisting of known pattern recognition (e.g., code vs. prose vs. documentation) and learning models. It may be more appropriate to use classification models such as decision trees, random forests, or CNNs. Or ideally, this would be a function of the SPT because domain relevance should be part of the classification system. Either way, the classification process would use a learning model so it would improve over time. 

Content will be labeled with its type (document, code, LLM response, user input, etc.) and a value rating as it relates to prompt usage:
- **Low value content**: Commands or system information
- **Medium value content**: Prose that is largely out of domain, documentation, generated code, or internal LLM deliberation
- **High value content**: User inputs and LLM's primary conversation and highly domain-relevant matter
- **Boosted content**: Content specifically tied to the prompt

### Content Storage

Every component of conversations would be stored in a data lake with appropriate metadata including the classification information.

### Content Retrieval

When a prompt is required, content will be retrieved from storage to form an ideal prompt.

### Query Generation

Query generation will be one of the primary learned functions for the SPT. The input that triggered the prompt would form the bulk of the query. Recent content from the same conversation and content with high relevance scores will be prioritized.

### Context Generation Process

1. An input prompt triggers the need for context.

2. The existing conversation will fill the draft context in reverse chronological order so that newer content makes it into the context, leaving out all low relevance content until the estimated token count reaches 75% (adjustable) of the max context window. Each piece of content will receive an attention score that is the combination of its temporal and context relevance along with prompt relevance boosting.

3. A query will be formed that will retrieve content from the data lake. The query will take the following into account:
   - Relevance to the prompt
   - The content's relevance score, excluding low relevance content
   - Temporal requirements from the prompt ("that thing from last week") will be included in the query

4. Each retrieved record from the query will be summarized and scored for attention. The level of summary will depend on the requirements as described in the prompt.

5. Retrieved content will be interleaved into the draft context by attention score, with the lower attention items dropping out of the context in order to maintain the 75% estimated context window.

## Training the SPTs

So how do you pretrain the SPTs? First, you would of course train them on domain-specific data which could include publicly available documents, copyrighted works for which you have the rights to, and inter-organizational data.

But that does not cover the ability to generate efficient context in light of infinite memory stores and recent conversation. For this, we would turn to the realm of LLMs. A professor once said, "All LLMs hallucinate, it's just that sometimes they're right." This is of course the problem of using LLMs to train other models. You can multiply the effects of hallucination. But this of course fails to note that humans are often wrong, and if you trained yourself, you would make your wrongness more wrong. We solve this problem through community conversation. At work, we get together with colleagues and experts. At school, we team with classmates and teachers. The same should apply to teaming LLMs.

### LLM Teaming

LLM Teaming is key to the training process. A set of different LLMs will form a team to judge and generate content for the training process. Their judgment function will use a voting system. For example, the LLM team may be asked whether the response to a prompt was correctly returned. Since an LLM may hallucinate, aligned voting would be used for the resulting judgment, with a hyperparameter set of the threshold of vote alignment.

There would be two phases to pretraining this model:

### Phase 1

Train the SPT on domain-specific data which could include publicly available documents, copyrighted works for which you have the rights to, and inter-organizational data.

### Phase 2

1. **Build Training Dataset**
   - Random chunks of domain-specific information would be exposed to the LLM team (paragraphs from authoritative texts, passages of laws or regulations, chapters of generally accepted practices, etc.)
   - Each LLM would be asked to generate a preset number of sample prompt and sample response pairs based on the exposed content
   - Each sample prompt and sample response pair would be reviewed by the LLM team through voting on whether it is a good pair, based on their understanding of language and the contents of the exposed data
   - Sample prompt and response pairs that pass muster will be added to the test dataset
   - This process will be repeated as often as necessary for the SPT to have sufficient sample data to achieve acceptable outputs

2. **SPT Training Process**
   - The SPT would be fed the sample prompt
   - The SPT would generate multiple candidate contexts based on the Context Generation Process above, varying the query inputs, relevance scores, and max estimated context size
   - The candidate contexts along with their utilized query and relevance parameters would be recorded along with the hyperparameter thresholds and the sample prompt and sample response
   - Each candidate context would be fed to each member of the LLM team, which would generate a candidate response
   - The candidate responses would be reviewed for accuracy to the sample response, generating a quality rating for the candidate prompt with an explanation
   - The LLM team members' quality rating and explanation would then be voted on by the LLM team to eliminate hallucinations
   - The resultant quality rating and explanation would be recorded alongside the sample context as mentioned above
   - The result data would be used to train and tune the model, thus the SPT becomes better at generating the correct domain-specific context through adjusting its relevance ratings and query parameters, while attempting to maintain a smaller context window for cost and speed

### Production Training

In production, as prompts are generated and the SPT generates context, the same training protocol above can be utilized to train the model using live data instead of training data. As this LLM teaming process will be expensive, training tests will periodically be performed to ensure the model is in tune.