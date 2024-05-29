/**
 * @file Pipelines provide a high-level, easy to use, API for running machine learning models.
 *
 * **Example:** Instantiate pipeline using the `pipeline` function.
 * ```javascript
 * import { pipeline } from '@xenova/transformers';
 *
 * const classifier = await pipeline('sentiment-analysis');
 * const output = await classifier('I love transformers!');
 * // [{'label': 'POSITIVE', 'score': 0.999817686}]
 * ```
 *
 * @module pipelines
 */

import {
    AutoTokenizer,
    PreTrainedTokenizer,
} from './tokenizers.js';
import {
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForQuestionAnswering,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    PreTrainedModel,
} from './models.js';
import {
    Processor
} from './processors.js';


import {
    Callable,
    dispatchCallback,
    product,
} from './utils/core.js';
import {
    softmax,
    max,
    getTopItems,
} from './utils/maths.js';
import {
    Tensor,
    mean_pooling,
    quantize_embeddings,
} from './utils/tensor.js';

/**
 * @callback DisposeType Disposes the item.
 * @returns {Promise<void>} A promise that resolves when the item has been disposed.
 *
 * @typedef {Object} Disposable
 * @property {DisposeType} dispose A promise that resolves when the pipeline has been disposed.
 */

/**
 * The Pipeline class is the class from which all pipelines inherit.
 * Refer to this class for methods shared across different pipelines.
 * @extends Callable
 */
export class Pipeline extends Callable {
    /**
     * Create a new Pipeline.
     * @param {Object} options An object containing the following properties:
     * @param {string} [options.task] The task of the pipeline. Useful for specifying subtasks.
     * @param {PreTrainedModel} [options.model] The model used by the pipeline.
     * @param {PreTrainedTokenizer} [options.tokenizer=null] The tokenizer used by the pipeline (if any).
     * @param {Processor} [options.processor=null] The processor used by the pipeline (if any).
     */
    constructor({ task, model, tokenizer = null, processor = null }) {
        super();
        this.task = task;
        this.model = model;
        this.tokenizer = tokenizer;
        this.processor = processor;
    }

    /** @type {DisposeType} */
    async dispose() {
        await this.model.dispose();
    }
}

/**
 * @typedef {Object} ModelTokenizerConstructorArgs
 * @property {string} task The task of the pipeline. Useful for specifying subtasks.
 * @property {PreTrainedModel} model The model used by the pipeline.
 * @property {PreTrainedTokenizer} tokenizer The tokenizer used by the pipeline.
 *
 * @typedef {ModelTokenizerConstructorArgs} TextPipelineConstructorArgs An object used to instantiate a text-based pipeline.
 */

/**
 * @typedef {Object} ModelProcessorConstructorArgs
 * @property {string} task The task of the pipeline. Useful for specifying subtasks.
 * @property {PreTrainedModel} model The model used by the pipeline.
 * @property {Processor} processor The processor used by the pipeline.
 */


/**
 * @typedef {Object} ModelTokenizerProcessorConstructorArgs
 * @property {string} task The task of the pipeline. Useful for specifying subtasks.
 * @property {PreTrainedModel} model The model used by the pipeline.
 * @property {PreTrainedTokenizer} tokenizer The tokenizer used by the pipeline.
 * @property {Processor} processor The processor used by the pipeline.
 *
 */

/**
 * @typedef {Object} TextClassificationSingle
 * @property {string} label The label predicted.
 * @property {number} score The corresponding probability.
 * @typedef {TextClassificationSingle[]} TextClassificationOutput
 *
 * @typedef {Object} TextClassificationPipelineOptions Parameters specific to text classification pipelines.
 * @property {number} [topk=1] The number of top predictions to be returned.
 *
 * @callback TextClassificationPipelineCallback Classify the text(s) given as inputs.
 * @param {string|string[]} texts The input text(s) to be classified.
 * @param {TextClassificationPipelineOptions} [options] The options to use for text classification.
 * @returns {Promise<TextClassificationOutput|TextClassificationOutput[]>} An array or object containing the predicted labels and scores.
 *
 * @typedef {TextPipelineConstructorArgs & TextClassificationPipelineCallback & Disposable} TextClassificationPipelineType
 */

/**
 * Text classification pipeline using any `ModelForSequenceClassification`.
 *
 * **Example:** Sentiment-analysis w/ `Xenova/distilbert-base-uncased-finetuned-sst-2-english`.
 * ```javascript
 * const classifier = await pipeline('sentiment-analysis', 'Xenova/distilbert-base-uncased-finetuned-sst-2-english');
 * const output = await classifier('I love transformers!');
 * // [{ label: 'POSITIVE', score: 0.999788761138916 }]
 * ```
 *
 * **Example:** Multilingual sentiment-analysis w/ `Xenova/bert-base-multilingual-uncased-sentiment` (and return top 5 classes).
 * ```javascript
 * const classifier = await pipeline('sentiment-analysis', 'Xenova/bert-base-multilingual-uncased-sentiment');
 * const output = await classifier('Le meilleur film de tous les temps.', { topk: 5 });
 * // [
 * //   { label: '5 stars', score: 0.9610759615898132 },
 * //   { label: '4 stars', score: 0.03323351591825485 },
 * //   { label: '3 stars', score: 0.0036155181005597115 },
 * //   { label: '1 star', score: 0.0011325967498123646 },
 * //   { label: '2 stars', score: 0.0009423971059732139 }
 * // ]
 * ```
 *
 * **Example:** Toxic comment classification w/ `Xenova/toxic-bert` (and return all classes).
 * ```javascript
 * const classifier = await pipeline('text-classification', 'Xenova/toxic-bert');
 * const output = await classifier('I hate you!', { topk: null });
 * // [
 * //   { label: 'toxic', score: 0.9593140482902527 },
 * //   { label: 'insult', score: 0.16187334060668945 },
 * //   { label: 'obscene', score: 0.03452680632472038 },
 * //   { label: 'identity_hate', score: 0.0223250575363636 },
 * //   { label: 'threat', score: 0.019197041168808937 },
 * //   { label: 'severe_toxic', score: 0.005651099607348442 }
 * // ]
 * ```
 */
export class TextClassificationPipeline extends (/** @type {new (options: TextPipelineConstructorArgs) => TextClassificationPipelineType} */ (Pipeline)) {

    /**
     * Create a new TextClassificationPipeline.
     * @param {TextPipelineConstructorArgs} options An object used to instantiate the pipeline.
     */
    constructor(options) {
        super(options);
    }

    /** @type {TextClassificationPipelineCallback} */
    async _call(texts, {
        topk = 1
    } = {}) {

        // Run tokenization
        const model_inputs = this.tokenizer(texts, {
            padding: true,
            truncation: true,
        });

        // Run model
        const outputs = await this.model(model_inputs)

        // TODO: Use softmax tensor function
        const function_to_apply =
            this.model.config.problem_type === 'multi_label_classification'
                ? batch => batch.sigmoid().data
                : batch => softmax(batch.data); // single_label_classification (default)

        const id2label = this.model.config.id2label;

        const toReturn = [];
        for (const batch of outputs.logits) {
            const output = function_to_apply(batch);
            const scores = getTopItems(output, topk);

            const vals = scores.map(x => ({
                label: id2label[x[0]],
                score: x[1],
            }));
            if (topk === 1) {
                toReturn.push(...vals);
            } else {
                toReturn.push(vals);
            }
        }

        return Array.isArray(texts) || topk === 1 ? /** @type {TextClassificationOutput} */ (toReturn) : /** @type {TextClassificationOutput[]} */ (toReturn)[0];
    }
}

/**
 * @typedef {Object} TokenClassificationSingle
 * @property {string} word The token/word classified. This is obtained by decoding the selected tokens.
 * @property {number} score The corresponding probability for `entity`.
 * @property {string} entity The entity predicted for that token/word.
 * @property {number} index The index of the corresponding token in the sentence.
 * @property {number} [start] The index of the start of the corresponding entity in the sentence.
 * @property {number} [end] The index of the end of the corresponding entity in the sentence.
 * @typedef {TokenClassificationSingle[]} TokenClassificationOutput
 *
 * @typedef {Object} TokenClassificationPipelineOptions Parameters specific to token classification pipelines.
 * @property {string[]} [ignore_labels] A list of labels to ignore.
 *
 * @callback TokenClassificationPipelineCallback Classify each token of the text(s) given as inputs.
 * @param {string|string[]} texts One or several texts (or one list of texts) for token classification.
 * @param {TokenClassificationPipelineOptions} [options] The options to use for token classification.
 * @returns {Promise<TokenClassificationOutput|TokenClassificationOutput[]>} The result.
 *
 * @typedef {TextPipelineConstructorArgs & TokenClassificationPipelineCallback & Disposable} TokenClassificationPipelineType
 */

/**
 * Named Entity Recognition pipeline using any `ModelForTokenClassification`.
 *
 * **Example:** Perform named entity recognition with `Xenova/bert-base-NER`.
 * ```javascript
 * const classifier = await pipeline('token-classification', 'Xenova/bert-base-NER');
 * const output = await classifier('My name is Sarah and I live in London');
 * // [
 * //   { entity: 'B-PER', score: 0.9980202913284302, index: 4, word: 'Sarah' },
 * //   { entity: 'B-LOC', score: 0.9994474053382874, index: 9, word: 'London' }
 * // ]
 * ```
 *
 * **Example:** Perform named entity recognition with `Xenova/bert-base-NER` (and return all labels).
 * ```javascript
 * const classifier = await pipeline('token-classification', 'Xenova/bert-base-NER');
 * const output = await classifier('Sarah lives in the United States of America', { ignore_labels: [] });
 * // [
 * //   { entity: 'B-PER', score: 0.9966587424278259, index: 1, word: 'Sarah' },
 * //   { entity: 'O', score: 0.9987385869026184, index: 2, word: 'lives' },
 * //   { entity: 'O', score: 0.9990072846412659, index: 3, word: 'in' },
 * //   { entity: 'O', score: 0.9988298416137695, index: 4, word: 'the' },
 * //   { entity: 'B-LOC', score: 0.9995510578155518, index: 5, word: 'United' },
 * //   { entity: 'I-LOC', score: 0.9990395307540894, index: 6, word: 'States' },
 * //   { entity: 'I-LOC', score: 0.9986724853515625, index: 7, word: 'of' },
 * //   { entity: 'I-LOC', score: 0.9975294470787048, index: 8, word: 'America' }
 * // ]
 * ```
 */
export class TokenClassificationPipeline extends (/** @type {new (options: TextPipelineConstructorArgs) => TokenClassificationPipelineType} */ (Pipeline)) {

    /**
     * Create a new TokenClassificationPipeline.
     * @param {TextPipelineConstructorArgs} options An object used to instantiate the pipeline.
     */
    constructor(options) {
        super(options);
    }

    /** @type {TokenClassificationPipelineCallback} */
    async _call(texts, {
        ignore_labels = ['O'],
    } = {}) {

        const isBatched = Array.isArray(texts);

        // Run tokenization
        const model_inputs = this.tokenizer(isBatched ? texts : [texts], {
            padding: true,
            truncation: true,
        });

        // Run model
        const outputs = await this.model(model_inputs)

        const logits = outputs.logits;
        const id2label = this.model.config.id2label;

        const toReturn = [];
        for (let i = 0; i < logits.dims[0]; ++i) {
            const ids = model_inputs.input_ids[i];
            const batch = logits[i];

            // List of tokens that aren't ignored
            const tokens = [];
            for (let j = 0; j < batch.dims[0]; ++j) {
                const tokenData = batch[j];
                const topScoreIndex = max(tokenData.data)[1];

                const entity = id2label ? id2label[topScoreIndex] : `LABEL_${topScoreIndex}`;
                if (ignore_labels.includes(entity)) {
                    // We predicted a token that should be ignored. So, we skip it.
                    continue;
                }

                // TODO add option to keep special tokens?
                const word = this.tokenizer.decode([ids[j].item()], { skip_special_tokens: true });
                if (word === '') {
                    // Was a special token. So, we skip it.
                    continue;
                }

                const scores = softmax(tokenData.data);

                tokens.push({
                    entity: entity,
                    score: scores[topScoreIndex],
                    index: j,
                    word: word,

                    // TODO: null for now, but will add
                    start: null,
                    end: null,
                });
            }
            toReturn.push(tokens);
        }
        return isBatched ? toReturn : toReturn[0];
    }
}

/**
 * @typedef {Object} QuestionAnsweringOutput
 * @property {number} score The probability associated to the answer.
 * @property {number} [start] The character start index of the answer (in the tokenized version of the input).
 * @property {number} [end] The character end index of the answer (in the tokenized version of the input).
 * @property {string} answer The answer to the question.
 *
 * @typedef {Object} QuestionAnsweringPipelineOptions Parameters specific to question answering pipelines.
 * @property {number} [topk=1] The number of top answer predictions to be returned.
 *
 * @callback QuestionAnsweringPipelineCallback Answer the question(s) given as inputs by using the context(s).
 * @param {string|string[]} question One or several question(s) (must be used in conjunction with the `context` argument).
 * @param {string|string[]} context One or several context(s) associated with the question(s) (must be used in conjunction with the `question` argument).
 * @param {QuestionAnsweringPipelineOptions} [options] The options to use for question answering.
 * @returns {Promise<QuestionAnsweringOutput|QuestionAnsweringOutput[]>} An array or object containing the predicted answers and scores.
 *
 * @typedef {TextPipelineConstructorArgs & QuestionAnsweringPipelineCallback & Disposable} QuestionAnsweringPipelineType
 */

/**
 * Question Answering pipeline using any `ModelForQuestionAnswering`.
 *
 * **Example:** Run question answering with `Xenova/distilbert-base-uncased-distilled-squad`.
 * ```javascript
 * const answerer = await pipeline('question-answering', 'Xenova/distilbert-base-uncased-distilled-squad');
 * const question = 'Who was Jim Henson?';
 * const context = 'Jim Henson was a nice puppet.';
 * const output = await answerer(question, context);
 * // {
 * //   answer: "a nice puppet",
 * //   score: 0.5768911502526741
 * // }
 * ```
 */
export class QuestionAnsweringPipeline extends (/** @type {new (options: TextPipelineConstructorArgs) => QuestionAnsweringPipelineType} */ (Pipeline)) {

    /**
     * Create a new QuestionAnsweringPipeline.
     * @param {TextPipelineConstructorArgs} options An object used to instantiate the pipeline.
     */
    constructor(options) {
        super(options);
    }

    /** @type {QuestionAnsweringPipelineCallback} */
    async _call(question, context, {
        topk = 1
    } = {}) {

        // Run tokenization
        const inputs = this.tokenizer(question, {
            text_pair: context,
            padding: true,
            truncation: true,
        });

        const output = await this.model(inputs);

        /** @type {QuestionAnsweringOutput[]} */
        const toReturn = [];
        for (let j = 0; j < output.start_logits.dims[0]; ++j) {
            const ids = inputs.input_ids[j];
            const sepIndex = ids.indexOf(this.tokenizer.sep_token_id);

            const s1 = Array.from(softmax(output.start_logits[j].data))
                .map((x, i) => [x, i])
                .filter(x => x[1] > sepIndex);
            const e1 = Array.from(softmax(output.end_logits[j].data))
                .map((x, i) => [x, i])
                .filter(x => x[1] > sepIndex);

            const options = product(s1, e1)
                .filter(x => x[0][1] <= x[1][1])
                .map(x => [x[0][1], x[1][1], x[0][0] * x[1][0]])
                .sort((a, b) => b[2] - a[2]);

            for (let k = 0; k < Math.min(options.length, topk); ++k) {
                const [start, end, score] = options[k];

                const answer_tokens = [...ids].slice(start, end + 1)

                const answer = this.tokenizer.decode(answer_tokens, {
                    skip_special_tokens: true,
                });

                // TODO add start and end?
                // NOTE: HF returns character index
                toReturn.push({
                    answer, score
                });
            }
        }

        // Mimic HF's return type based on topk
        return (topk === 1) ? toReturn[0] : toReturn;
    }
}


/**
 * @typedef {Object} FillMaskSingle
 * @property {string} sequence The corresponding input with the mask token prediction.
 * @property {number} score The corresponding probability.
 * @property {number} token The predicted token id (to replace the masked one).
 * @property {string} token_str The predicted token (to replace the masked one).
 * @typedef {FillMaskSingle[]} FillMaskOutput
 *
 * @typedef {Object} FillMaskPipelineOptions Parameters specific to fill mask pipelines.
 * @property {number} [topk=5] When passed, overrides the number of predictions to return.
 *
 * @callback FillMaskPipelineCallback Fill the masked token in the text(s) given as inputs.
 * @param {string|string[]} texts One or several texts (or one list of prompts) with masked tokens.
 * @param {FillMaskPipelineOptions} [options] The options to use for masked language modelling.
 * @returns {Promise<FillMaskOutput|FillMaskOutput[]>} An array of objects containing the score, predicted token, predicted token string,
 * and the sequence with the predicted token filled in, or an array of such arrays (one for each input text).
 * If only one input text is given, the output will be an array of objects.
 * @throws {Error} When the mask token is not found in the input text.
 *
 * @typedef {TextPipelineConstructorArgs & FillMaskPipelineCallback & Disposable} FillMaskPipelineType
 */

/**
 * Masked language modeling prediction pipeline using any `ModelWithLMHead`.
 *
 * **Example:** Perform masked language modelling (a.k.a. "fill-mask") with `Xenova/bert-base-uncased`.
 * ```javascript
 * const unmasker = await pipeline('fill-mask', 'Xenova/bert-base-cased');
 * const output = await unmasker('The goal of life is [MASK].');
 * // [
 * //   { token_str: 'survival', score: 0.06137419492006302, token: 8115, sequence: 'The goal of life is survival.' },
 * //   { token_str: 'love', score: 0.03902450203895569, token: 1567, sequence: 'The goal of life is love.' },
 * //   { token_str: 'happiness', score: 0.03253183513879776, token: 9266, sequence: 'The goal of life is happiness.' },
 * //   { token_str: 'freedom', score: 0.018736306577920914, token: 4438, sequence: 'The goal of life is freedom.' },
 * //   { token_str: 'life', score: 0.01859794743359089, token: 1297, sequence: 'The goal of life is life.' }
 * // ]
 * ```
 *
 * **Example:** Perform masked language modelling (a.k.a. "fill-mask") with `Xenova/bert-base-cased` (and return top result).
 * ```javascript
 * const unmasker = await pipeline('fill-mask', 'Xenova/bert-base-cased');
 * const output = await unmasker('The Milky Way is a [MASK] galaxy.', { topk: 1 });
 * // [{ token_str: 'spiral', score: 0.6299987435340881, token: 14061, sequence: 'The Milky Way is a spiral galaxy.' }]
 * ```
 */
export class FillMaskPipeline extends (/** @type {new (options: TextPipelineConstructorArgs) => FillMaskPipelineType} */ (Pipeline)) {

    /**
     * Create a new FillMaskPipeline.
     * @param {TextPipelineConstructorArgs} options An object used to instantiate the pipeline.
     */
    constructor(options) {
        super(options);
    }

    /** @type {FillMaskPipelineCallback} */
    async _call(texts, {
        topk = 5
    } = {}) {

        // Run tokenization
        const model_inputs = this.tokenizer(texts, {
            padding: true,
            truncation: true,
        });

        // Run model
        const outputs = await this.model(model_inputs)

        const toReturn = [];

        for (let i = 0; i < model_inputs.input_ids.dims[0]; ++i) {
            const ids = model_inputs.input_ids[i];
            const mask_token_index = ids.indexOf(this.tokenizer.mask_token_id)

            if (mask_token_index === -1) {
                throw Error(`Mask token (${this.tokenizer.mask_token}) not found in text.`)
            }
            const logits = outputs.logits[i];
            const itemLogits = logits[mask_token_index];

            const scores = getTopItems(softmax(itemLogits.data), topk);

            toReturn.push(scores.map(x => {
                const sequence = [...ids];
                sequence[mask_token_index] = x[0];

                return {
                    score: x[1],
                    token: x[0],
                    token_str: this.tokenizer.model.vocab[x[0]],
                    sequence: this.tokenizer.decode(sequence, { skip_special_tokens: true }),
                }
            }));
        }
        return Array.isArray(texts) ? toReturn : toReturn[0];
    }
}


/**
 * @typedef {Object} Text2TextGenerationSingle
 * @property {string} generated_text The generated text.
 * @typedef {Text2TextGenerationSingle[]} Text2TextGenerationOutput
 *
 * @callback Text2TextGenerationPipelineCallback Generate the output text(s) using text(s) given as inputs.
 * @param {string|string[]} texts Input text for the encoder.
 * @param {import('./utils/generation.js').GenerationConfigType} [options] Additional keyword arguments to pass along to the generate method of the model.
 * @returns {Promise<Text2TextGenerationOutput|Text2TextGenerationOutput[]>}
 *
 * @typedef {TextPipelineConstructorArgs & Text2TextGenerationPipelineCallback & Disposable} Text2TextGenerationPipelineType
 */

/**
 * Text2TextGenerationPipeline class for generating text using a model that performs text-to-text generation tasks.
 *
 * **Example:** Text-to-text generation w/ `Xenova/LaMini-Flan-T5-783M`.
 * ```javascript
 * const generator = await pipeline('text2text-generation', 'Xenova/LaMini-Flan-T5-783M');
 * const output = await generator('how can I become more healthy?', {
 *   max_new_tokens: 100,
 * });
 * // [{ generated_text: "To become more healthy, you can: 1. Eat a balanced diet with plenty of fruits, vegetables, whole grains, lean proteins, and healthy fats. 2. Stay hydrated by drinking plenty of water. 3. Get enough sleep and manage stress levels. 4. Avoid smoking and excessive alcohol consumption. 5. Regularly exercise and maintain a healthy weight. 6. Practice good hygiene and sanitation. 7. Seek medical attention if you experience any health issues." }]
 * ```
 */
export class Text2TextGenerationPipeline extends (/** @type {new (options: TextPipelineConstructorArgs) => Text2TextGenerationPipelineType} */ (Pipeline)) {
    /** @type {'generated_text'} */
    _key = 'generated_text';

    /**
     * Create a new Text2TextGenerationPipeline.
     * @param {TextPipelineConstructorArgs} options An object used to instantiate the pipeline.
     */
    constructor(options) {
        super(options);
    }

    /** @type {Text2TextGenerationPipelineCallback} */
    async _call(texts, generate_kwargs = {}) {
        if (!Array.isArray(texts)) {
            texts = [texts];
        }


        // Add global prefix, if present
        if (this.model.config.prefix) {
            texts = texts.map(x => this.model.config.prefix + x)
        }

        // Handle task specific params:
        const task_specific_params = this.model.config.task_specific_params
        if (task_specific_params && task_specific_params[this.task]) {
            // Add prefixes, if present
            if (task_specific_params[this.task].prefix) {
                texts = texts.map(x => task_specific_params[this.task].prefix + x)
            }

            // TODO update generation config
        }

        const tokenizer = this.tokenizer;
        const tokenizer_options = {
            padding: true,
            truncation: true,
        }
        let input_ids;
        if (this instanceof TranslationPipeline && '_build_translation_inputs' in tokenizer) {
            // TODO: move to Translation pipeline?
            // Currently put here to avoid code duplication
            // @ts-ignore
            input_ids = tokenizer._build_translation_inputs(texts, tokenizer_options, generate_kwargs).input_ids;

        } else {
            input_ids = tokenizer(texts, tokenizer_options).input_ids;
        }

        const outputTokenIds = await this.model.generate(input_ids, generate_kwargs);

        return tokenizer.batch_decode(outputTokenIds, {
            skip_special_tokens: true,
        }).map(text => ({ [this._key]: text }));
    }
}


/**
 * @typedef {Object} SummarizationSingle
 * @property {string} summary_text The summary text.
 * @typedef {SummarizationSingle[]} SummarizationOutput
 *
 * @callback SummarizationPipelineCallback Summarize the text(s) given as inputs.
 * @param {string|string[]} texts One or several articles (or one list of articles) to summarize.
 * @param {import('./utils/generation.js').GenerationConfigType} [options] Additional keyword arguments to pass along to the generate method of the model.
 * @returns {Promise<SummarizationOutput|SummarizationOutput[]>}
 *
 * @typedef {TextPipelineConstructorArgs & SummarizationPipelineCallback & Disposable} SummarizationPipelineType
 */

/**
 * A pipeline for summarization tasks, inheriting from Text2TextGenerationPipeline.
 *
 * **Example:** Summarization w/ `Xenova/distilbart-cnn-6-6`.
 * ```javascript
 * const generator = await pipeline('summarization', 'Xenova/distilbart-cnn-6-6');
 * const text = 'The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, ' +
 *   'and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. ' +
 *   'During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest ' +
 *   'man-made structure in the world, a title it held for 41 years until the Chrysler Building in New ' +
 *   'York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to ' +
 *   'the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the ' +
 *   'Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second ' +
 *   'tallest free-standing structure in France after the Millau Viaduct.';
 * const output = await generator(text, {
 *   max_new_tokens: 100,
 * });
 * // [{ summary_text: ' The Eiffel Tower is about the same height as an 81-storey building and the tallest structure in Paris. It is the second tallest free-standing structure in France after the Millau Viaduct.' }]
 * ```
 */
export class SummarizationPipeline extends (/** @type {new (options: TextPipelineConstructorArgs) => SummarizationPipelineType} */ (/** @type {any} */ (Text2TextGenerationPipeline))) {
    /** @type {'summary_text'} */
    _key = 'summary_text';

    /**
     * Create a new SummarizationPipeline.
     * @param {TextPipelineConstructorArgs} options An object used to instantiate the pipeline.
     */
    constructor(options) {
        super(options);
    }
}


/**
 * @typedef {Object} TranslationSingle
 * @property {string} translation_text The translated text.
 * @typedef {TranslationSingle[]} TranslationOutput
 *
 * @callback TranslationPipelineCallback Translate the text(s) given as inputs.
 * @param {string|string[]} texts Texts to be translated.
 * @param {import('./utils/generation.js').GenerationConfigType} [options] Additional keyword arguments to pass along to the generate method of the model.
 * @returns {Promise<TranslationOutput|TranslationOutput[]>}
 *
 * @typedef {TextPipelineConstructorArgs & TranslationPipelineCallback & Disposable} TranslationPipelineType
 */

/**
 * Translates text from one language to another.
 *
 * **Example:** Multilingual translation w/ `Xenova/nllb-200-distilled-600M`.
 *
 * See [here](https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200)
 * for the full list of languages and their corresponding codes.
 *
 * ```javascript
 * const translator = await pipeline('translation', 'Xenova/nllb-200-distilled-600M');
 * const output = await translator('जीवन एक चॉकलेट बॉक्स की तरह है।', {
 *   src_lang: 'hin_Deva', // Hindi
 *   tgt_lang: 'fra_Latn', // French
 * });
 * // [{ translation_text: 'La vie est comme une boîte à chocolat.' }]
 * ```
 *
 * **Example:** Multilingual translation w/ `Xenova/m2m100_418M`.
 *
 * See [here](https://huggingface.co/facebook/m2m100_418M#languages-covered)
 * for the full list of languages and their corresponding codes.
 *
 * ```javascript
 * const translator = await pipeline('translation', 'Xenova/m2m100_418M');
 * const output = await translator('生活就像一盒巧克力。', {
 *   src_lang: 'zh', // Chinese
 *   tgt_lang: 'en', // English
 * });
 * // [{ translation_text: 'Life is like a box of chocolate.' }]
 * ```
 *
 * **Example:** Multilingual translation w/ `Xenova/mbart-large-50-many-to-many-mmt`.
 *
 * See [here](https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt#languages-covered)
 * for the full list of languages and their corresponding codes.
 *
 * ```javascript
 * const translator = await pipeline('translation', 'Xenova/mbart-large-50-many-to-many-mmt');
 * const output = await translator('संयुक्त राष्ट्र के प्रमुख का कहना है कि सीरिया में कोई सैन्य समाधान नहीं है', {
 *   src_lang: 'hi_IN', // Hindi
 *   tgt_lang: 'fr_XX', // French
 * });
 * // [{ translation_text: 'Le chef des Nations affirme qu 'il n 'y a military solution in Syria.' }]
 * ```
 */
export class TranslationPipeline extends (/** @type {new (options: TextPipelineConstructorArgs) => TranslationPipelineType} */ (/** @type {any} */ (Text2TextGenerationPipeline))) {
    /** @type {'translation_text'} */
    _key = 'translation_text';

    /**
     * Create a new TranslationPipeline.
     * @param {TextPipelineConstructorArgs} options An object used to instantiate the pipeline.
     */
    constructor(options) {
        super(options);
    }
}

function isChat(x) {
    return Array.isArray(x) && x.every(x => 'role' in x && 'content' in x);
}

/**
 * @typedef {import('./tokenizers.js').Message[]} Chat
 *
 * @typedef {Object} TextGenerationSingle
 * @property {string|Chat} generated_text The generated text.
 * @typedef {TextGenerationSingle[]} TextGenerationOutput
 *
 * @typedef {Object} TextGenerationSpecificParams Parameters specific to text-generation pipelines.
 * @property {boolean} [add_special_tokens] Whether or not to add special tokens when tokenizing the sequences.
 * @property {boolean} [return_full_text=true] If set to `false` only added text is returned, otherwise the full text is returned.
 * @typedef {import('./utils/generation.js').GenerationConfigType & TextGenerationSpecificParams} TextGenerationConfig
 *
 * @callback TextGenerationPipelineCallback Complete the prompt(s) given as inputs.
 * @param {string|string[]|Chat|Chat[]} texts One or several prompts (or one list of prompts) to complete.
 * @param {TextGenerationConfig} [options] Additional keyword arguments to pass along to the generate method of the model.
 * @returns {Promise<TextGenerationOutput|TextGenerationOutput[]>} An array or object containing the generated texts.
 *
 * @typedef {TextPipelineConstructorArgs & TextGenerationPipelineCallback & Disposable} TextGenerationPipelineType
 */

/**
 * Language generation pipeline using any `ModelWithLMHead` or `ModelForCausalLM`.
 * This pipeline predicts the words that will follow a specified text prompt.
 * NOTE: For the full list of generation parameters, see [`GenerationConfig`](./utils/generation#module_utils/generation.GenerationConfig).
 *
 * **Example:** Text generation with `Xenova/distilgpt2` (default settings).
 * ```javascript
 * const generator = await pipeline('text-generation', 'Xenova/distilgpt2');
 * const text = 'I enjoy walking with my cute dog,';
 * const output = await generator(text);
 * // [{ generated_text: "I enjoy walking with my cute dog, and I love to play with the other dogs." }]
 * ```
 *
 * **Example:** Text generation with `Xenova/distilgpt2` (custom settings).
 * ```javascript
 * const generator = await pipeline('text-generation', 'Xenova/distilgpt2');
 * const text = 'Once upon a time, there was';
 * const output = await generator(text, {
 *   temperature: 2,
 *   max_new_tokens: 10,
 *   repetition_penalty: 1.5,
 *   no_repeat_ngram_size: 2,
 *   num_beams: 2,
 *   num_return_sequences: 2,
 * });
 * // [{
 * //   "generated_text": "Once upon a time, there was an abundance of information about the history and activities that"
 * // }, {
 * //   "generated_text": "Once upon a time, there was an abundance of information about the most important and influential"
 * // }]
 * ```
 *
 * **Example:** Run code generation with `Xenova/codegen-350M-mono`.
 * ```javascript
 * const generator = await pipeline('text-generation', 'Xenova/codegen-350M-mono');
 * const text = 'def fib(n):';
 * const output = await generator(text, {
 *   max_new_tokens: 44,
 * });
 * // [{
 * //   generated_text: 'def fib(n):\n' +
 * //     '    if n == 0:\n' +
 * //     '        return 0\n' +
 * //     '    elif n == 1:\n' +
 * //     '        return 1\n' +
 * //     '    else:\n' +
 * //     '        return fib(n-1) + fib(n-2)\n'
 * // }]
 * ```
 */
export class TextGenerationPipeline extends (/** @type {new (options: TextPipelineConstructorArgs) => TextGenerationPipelineType} */ (Pipeline)) {

    /**
     * Create a new TextGenerationPipeline.
     * @param {TextPipelineConstructorArgs} options An object used to instantiate the pipeline.
     */
    constructor(options) {
        super(options);
    }

    /** @type {TextGenerationPipelineCallback} */
    async _call(texts, generate_kwargs = {}) {
        let isBatched = false;
        let isChatInput = false;

        // Normalize inputs
        /** @type {string[]} */
        let inputs;
        if (typeof texts === 'string') {
            inputs = texts = [texts];
        } else if (Array.isArray(texts) && texts.every(x => typeof x === 'string')) {
            isBatched = true;
            inputs = /** @type {string[]} */(texts);
        } else {
            if (isChat(texts)) {
                texts = [/** @type {Chat} */(texts)];
            } else if (Array.isArray(texts) && texts.every(isChat)) {
                isBatched = true;
            } else {
                throw new Error('Input must be a string, an array of strings, a Chat, or an array of Chats');
            }
            isChatInput = true;

            // If the input is a chat, we need to apply the chat template
            inputs = /** @type {string[]} */(/** @type {Chat[]} */ (texts).map(
                x => this.tokenizer.apply_chat_template(x, {
                    tokenize: false,
                    add_generation_prompt: true,
                })
            ));
        }

        // By default, do not add special tokens
        const add_special_tokens = generate_kwargs.add_special_tokens ?? false;

        // By default, return full text
        const return_full_text = isChatInput
            ? false
            : generate_kwargs.return_full_text ?? true;

        this.tokenizer.padding_side = 'left';
        const { input_ids, attention_mask } = this.tokenizer(inputs, {
            add_special_tokens,
            padding: true,
            truncation: true,
        });

        const outputTokenIds = await this.model.generate(input_ids, generate_kwargs, null, {
            inputs_attention_mask: attention_mask
        });

        let decoded = this.tokenizer.batch_decode(outputTokenIds, {
            skip_special_tokens: true,
        });


        let promptLengths;
        if (!return_full_text && input_ids.dims.at(-1) > 0) {
            promptLengths = this.tokenizer.batch_decode(input_ids, {
                skip_special_tokens: true,
            }).map(x => x.length);
        }

        /** @type {TextGenerationOutput[]} */
        const toReturn = Array.from({ length: texts.length }, _ => []);
        for (let i = 0; i < decoded.length; ++i) {
            const textIndex = Math.floor(i / outputTokenIds.length * texts.length);

            if (promptLengths) {
                // Trim the decoded text to only include the generated part
                decoded[i] = decoded[i].slice(promptLengths[textIndex]);
            }
            toReturn[textIndex].push({
                generated_text: isChatInput
                    ? [
                        ...((/** @type {Chat[]} */(texts)[textIndex])),
                        { role: 'assistant', content: decoded[i] },
                    ]
                    : decoded[i]
            });
        }
        return (!isBatched && toReturn.length === 1) ? toReturn[0] : toReturn;
    }
}

/**
 * @typedef {Object} ZeroShotClassificationOutput
 * @property {string} sequence The sequence for which this is the output.
 * @property {string[]} labels The labels sorted by order of likelihood.
 * @property {number[]} scores The probabilities for each of the labels.
 *
 * @typedef {Object} ZeroShotClassificationPipelineOptions Parameters specific to zero-shot classification pipelines.
 * @property {string} [hypothesis_template="This example is {}."] The template used to turn each
 * candidate label into an NLI-style hypothesis. The candidate label will replace the {} placeholder.
 * @property {boolean} [multi_label=false] Whether or not multiple candidate labels can be true.
 * If `false`, the scores are normalized such that the sum of the label likelihoods for each sequence
 * is 1. If `true`, the labels are considered independent and probabilities are normalized for each
 * candidate by doing a softmax of the entailment score vs. the contradiction score.
 *
 * @callback ZeroShotClassificationPipelineCallback Classify the sequence(s) given as inputs.
 * @param {string|string[]} texts The sequence(s) to classify, will be truncated if the model input is too large.
 * @param {string|string[]} candidate_labels The set of possible class labels to classify each sequence into.
 * Can be a single label, a string of comma-separated labels, or a list of labels.
 * @param {ZeroShotClassificationPipelineOptions} [options] The options to use for zero-shot classification.
 * @returns {Promise<ZeroShotClassificationOutput|ZeroShotClassificationOutput[]>} An array or object containing the predicted labels and scores.
 *
 * @typedef {TextPipelineConstructorArgs & ZeroShotClassificationPipelineCallback & Disposable} ZeroShotClassificationPipelineType
 */

/**
 * NLI-based zero-shot classification pipeline using a `ModelForSequenceClassification`
 * trained on NLI (natural language inference) tasks. Equivalent of `text-classification`
 * pipelines, but these models don't require a hardcoded number of potential classes, they
 * can be chosen at runtime. It usually means it's slower but it is **much** more flexible.
 *
 * **Example:** Zero shot classification with `Xenova/mobilebert-uncased-mnli`.
 * ```javascript
 * const classifier = await pipeline('zero-shot-classification', 'Xenova/mobilebert-uncased-mnli');
 * const text = 'Last week I upgraded my iOS version and ever since then my phone has been overheating whenever I use your app.';
 * const labels = [ 'mobile', 'billing', 'website', 'account access' ];
 * const output = await classifier(text, labels);
 * // {
 * //   sequence: 'Last week I upgraded my iOS version and ever since then my phone has been overheating whenever I use your app.',
 * //   labels: [ 'mobile', 'website', 'billing', 'account access' ],
 * //   scores: [ 0.5562091040482018, 0.1843621307860853, 0.13942646639336376, 0.12000229877234923 ]
 * // }
 * ```
 *
 * **Example:** Zero shot classification with `Xenova/nli-deberta-v3-xsmall` (multi-label).
 * ```javascript
 * const classifier = await pipeline('zero-shot-classification', 'Xenova/nli-deberta-v3-xsmall');
 * const text = 'I have a problem with my iphone that needs to be resolved asap!';
 * const labels = [ 'urgent', 'not urgent', 'phone', 'tablet', 'computer' ];
 * const output = await classifier(text, labels, { multi_label: true });
 * // {
 * //   sequence: 'I have a problem with my iphone that needs to be resolved asap!',
 * //   labels: [ 'urgent', 'phone', 'computer', 'tablet', 'not urgent' ],
 * //   scores: [ 0.9958870956360275, 0.9923963400697035, 0.002333537946160235, 0.0015134138567598765, 0.0010699384208377163 ]
 * // }
 * ```
 */
export class ZeroShotClassificationPipeline extends (/** @type {new (options: TextPipelineConstructorArgs) => ZeroShotClassificationPipelineType} */ (Pipeline)) {
    /**
     * Create a new ZeroShotClassificationPipeline.
     * @param {TextPipelineConstructorArgs} options An object used to instantiate the pipeline.
     */
    constructor(options) {
        super(options);

        // Use model config to get label2id mapping
        this.label2id = Object.fromEntries(
            Object.entries((/** @type {any} */(this).model).config.label2id).map(
                ([k, v]) => [k.toLowerCase(), v]
            )
        );

        this.entailment_id = this.label2id['entailment'];
        if (this.entailment_id === undefined) {
            console.warn("Could not find 'entailment' in label2id mapping. Using 2 as entailment_id.");
            this.entailment_id = 2;
        }

        this.contradiction_id = this.label2id['contradiction'] ?? this.label2id['not_entailment'];
        if (this.contradiction_id === undefined) {
            console.warn("Could not find 'contradiction' in label2id mapping. Using 0 as contradiction_id.");
            this.contradiction_id = 0;
        }
    }

    /** @type {ZeroShotClassificationPipelineCallback} */
    async _call(texts, candidate_labels, {
        hypothesis_template = "This example is {}.",
        multi_label = false,
    } = {}) {

        const isBatched = Array.isArray(texts);
        if (!isBatched) {
            texts = [/** @type {string} */ (texts)];
        }
        if (!Array.isArray(candidate_labels)) {
            candidate_labels = [candidate_labels];
        }

        // Insert labels into hypothesis template
        const hypotheses = candidate_labels.map(
            x => hypothesis_template.replace('{}', x)
        );

        // How to perform the softmax over the logits:
        //  - true:  softmax over the entailment vs. contradiction dim for each label independently
        //  - false: softmax the "entailment" logits over all candidate labels
        const softmaxEach = multi_label || candidate_labels.length === 1;

        /** @type {ZeroShotClassificationOutput[]} */
        const toReturn = [];
        for (const premise of texts) {
            const entails_logits = [];

            for (const hypothesis of hypotheses) {
                const inputs = this.tokenizer(premise, {
                    text_pair: hypothesis,
                    padding: true,
                    truncation: true,
                })
                const outputs = await this.model(inputs)

                if (softmaxEach) {
                    entails_logits.push([
                        outputs.logits.data[this.contradiction_id],
                        outputs.logits.data[this.entailment_id]
                    ])
                } else {
                    entails_logits.push(outputs.logits.data[this.entailment_id])
                }
            }

            /** @type {number[]} */
            const scores = softmaxEach
                ? entails_logits.map(x => softmax(x)[1])
                : softmax(entails_logits);

            // Sort by scores (desc) and return scores with indices
            const scores_sorted = scores
                .map((x, i) => [x, i])
                .sort((a, b) => (b[0] - a[0]));

            toReturn.push({
                sequence: premise,
                labels: scores_sorted.map(x => candidate_labels[x[1]]),
                scores: scores_sorted.map(x => x[0]),
            });
        }
        return isBatched ? toReturn : toReturn[0];
    }
}

/**
 * @typedef {Object} FeatureExtractionPipelineOptions Parameters specific to feature extraction pipelines.
 * @property {'none'|'mean'|'cls'} [pooling="none"] The pooling method to use.
 * @property {boolean} [normalize=false] Whether or not to normalize the embeddings in the last dimension.
 * @property {boolean} [quantize=false] Whether or not to quantize the embeddings.
 * @property {'binary'|'ubinary'} [precision='binary'] The precision to use for quantization.
 *
 * @callback FeatureExtractionPipelineCallback Extract the features of the input(s).
 * @param {string|string[]} texts One or several texts (or one list of texts) to get the features of.
 * @param {FeatureExtractionPipelineOptions} [options] The options to use for feature extraction.
 * @returns {Promise<Tensor>} The features computed by the model.
 *
 * @typedef {TextPipelineConstructorArgs & FeatureExtractionPipelineCallback & Disposable} FeatureExtractionPipelineType
 */

/**
 * Feature extraction pipeline using no model head. This pipeline extracts the hidden
 * states from the base transformer, which can be used as features in downstream tasks.
 *
 * **Example:** Run feature extraction with `bert-base-uncased` (without pooling/normalization).
 * ```javascript
 * const extractor = await pipeline('feature-extraction', 'Xenova/bert-base-uncased', { revision: 'default' });
 * const output = await extractor('This is a simple test.');
 * // Tensor {
 * //   type: 'float32',
 * //   data: Float32Array [0.05939924716949463, 0.021655935794115067, ...],
 * //   dims: [1, 8, 768]
 * // }
 * ```
 *
 * **Example:** Run feature extraction with `bert-base-uncased` (with pooling/normalization).
 * ```javascript
 * const extractor = await pipeline('feature-extraction', 'Xenova/bert-base-uncased', { revision: 'default' });
 * const output = await extractor('This is a simple test.', { pooling: 'mean', normalize: true });
 * // Tensor {
 * //   type: 'float32',
 * //   data: Float32Array [0.03373778983950615, -0.010106077417731285, ...],
 * //   dims: [1, 768]
 * // }
 * ```
 *
 * **Example:** Calculating embeddings with `sentence-transformers` models.
 * ```javascript
 * const extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
 * const output = await extractor('This is a simple test.', { pooling: 'mean', normalize: true });
 * // Tensor {
 * //   type: 'float32',
 * //   data: Float32Array [0.09094982594251633, -0.014774246141314507, ...],
 * //   dims: [1, 384]
 * // }
 * ```
 * **Example:** Calculating binary embeddings with `sentence-transformers` models.
 * ```javascript
 * const extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
 * const output = await extractor('This is a simple test.', { pooling: 'mean', quantize: true, precision: 'binary' });
 * // Tensor {
 * //   type: 'int8',
 * //   data: Int8Array [49, 108, 24, ...],
 * //   dims: [1, 48]
 * // }
 * ```
 */
export class FeatureExtractionPipeline extends (/** @type {new (options: TextPipelineConstructorArgs) => FeatureExtractionPipelineType} */ (Pipeline)) {
    /**
     * Create a new FeatureExtractionPipeline.
     * @param {TextPipelineConstructorArgs} options An object used to instantiate the pipeline.
     */
    constructor(options) {
        super(options);
    }

    /** @type {FeatureExtractionPipelineCallback} */
    async _call(texts, {
        pooling = /** @type {'none'} */('none'),
        normalize = false,
        quantize = false,
        precision = /** @type {'binary'} */('binary'),
    } = {}) {

        // Run tokenization
        const model_inputs = this.tokenizer(texts, {
            padding: true,
            truncation: true,
        });

        // Run model
        const outputs = await this.model(model_inputs)

        // TODO: Provide warning to the user that they might be using model which was not exported
        // specifically for feature extraction
        // console.log(this.model.config)
        // console.log(outputs)

        /** @type {Tensor} */
        let result = outputs.last_hidden_state ?? outputs.logits ?? outputs.token_embeddings;
        if (pooling === 'none') {
            // Skip pooling
        } else if (pooling === 'mean') {
            result = mean_pooling(result, model_inputs.attention_mask);
        } else if (pooling === 'cls') {
            result = result.slice(null, 0);
        } else {
            throw Error(`Pooling method '${pooling}' not supported.`);
        }

        if (normalize) {
            result = result.normalize(2, -1);
        }

        if (quantize) {
            result = quantize_embeddings(result, precision);
        }

        return result;
    }
}

const SUPPORTED_TASKS = Object.freeze({
    "text-classification": {
        "tokenizer": AutoTokenizer,
        "pipeline": TextClassificationPipeline,
        "model": AutoModelForSequenceClassification,
        "default": {
            // TODO: replace with original
            // "model": "distilbert-base-uncased-finetuned-sst-2-english",
            "model": "Xenova/distilbert-base-uncased-finetuned-sst-2-english",
        },
        "type": "text",
    },
    "token-classification": {
        "tokenizer": AutoTokenizer,
        "pipeline": TokenClassificationPipeline,
        "model": AutoModelForTokenClassification,
        "default": {
            // TODO: replace with original
            // "model": "Davlan/bert-base-multilingual-cased-ner-hrl",
            "model": "Xenova/bert-base-multilingual-cased-ner-hrl",
        },
        "type": "text",
    },
    "question-answering": {
        "tokenizer": AutoTokenizer,
        "pipeline": QuestionAnsweringPipeline,
        "model": AutoModelForQuestionAnswering,
        "default": {
            // TODO: replace with original
            // "model": "distilbert-base-cased-distilled-squad",
            "model": "Xenova/distilbert-base-cased-distilled-squad",
        },
        "type": "text",
    },

    "fill-mask": {
        "tokenizer": AutoTokenizer,
        "pipeline": FillMaskPipeline,
        "model": AutoModelForMaskedLM,
        "default": {
            // TODO: replace with original
            // "model": "bert-base-uncased",
            "model": "Xenova/bert-base-uncased",
        },
        "type": "text",
    },
    "summarization": {
        "tokenizer": AutoTokenizer,
        "pipeline": SummarizationPipeline,
        "model": AutoModelForSeq2SeqLM,
        "default": {
            // TODO: replace with original
            // "model": "sshleifer/distilbart-cnn-6-6",
            "model": "Xenova/distilbart-cnn-6-6",
        },
        "type": "text",
    },
    "translation": {
        "tokenizer": AutoTokenizer,
        "pipeline": TranslationPipeline,
        "model": AutoModelForSeq2SeqLM,
        "default": {
            // TODO: replace with original
            // "model": "t5-small",
            "model": "Xenova/t5-small",
        },
        "type": "text",
    },
    "text2text-generation": {
        "tokenizer": AutoTokenizer,
        "pipeline": Text2TextGenerationPipeline,
        "model": AutoModelForSeq2SeqLM,
        "default": {
            // TODO: replace with original
            // "model": "google/flan-t5-small",
            "model": "Xenova/flan-t5-small",
        },
        "type": "text",
    },
    "text-generation": {
        "tokenizer": AutoTokenizer,
        "pipeline": TextGenerationPipeline,
        "model": AutoModelForCausalLM,
        "default": {
            // TODO: replace with original
            // "model": "gpt2",
            "model": "Xenova/gpt2",
        },
        "type": "text",
    },
    "zero-shot-classification": {
        "tokenizer": AutoTokenizer,
        "pipeline": ZeroShotClassificationPipeline,
        "model": AutoModelForSequenceClassification,
        "default": {
            // TODO: replace with original
            // "model": "typeform/distilbert-base-uncased-mnli",
            "model": "Xenova/distilbert-base-uncased-mnli",
        },
        "type": "text",
    },

    // This task serves as a useful interface for dealing with sentence-transformers (https://huggingface.co/sentence-transformers).
    "feature-extraction": {
        "tokenizer": AutoTokenizer,
        "pipeline": FeatureExtractionPipeline,
        "model": AutoModel,
        "default": {
            // TODO: replace with original
            // "model": "sentence-transformers/all-MiniLM-L6-v2",
            "model": "Xenova/all-MiniLM-L6-v2",
        },
        "type": "text",
    },
})


// TODO: Add types for TASK_ALIASES
const TASK_ALIASES = Object.freeze({
    "sentiment-analysis": "text-classification",
    "ner": "token-classification",
    // "vqa": "visual-question-answering", // TODO: Add
    "asr": "automatic-speech-recognition",

    // Add for backwards compatibility
    "embeddings": "feature-extraction",
});

/**
 * @typedef {keyof typeof SUPPORTED_TASKS} TaskType
 * @typedef {keyof typeof TASK_ALIASES} AliasType
 * @typedef {TaskType | AliasType} PipelineType All possible pipeline types.
 * @typedef {{[K in TaskType]: InstanceType<typeof SUPPORTED_TASKS[K]["pipeline"]>}} SupportedTasks A mapping of pipeline names to their corresponding pipeline classes.
 * @typedef {{[K in AliasType]: InstanceType<typeof SUPPORTED_TASKS[TASK_ALIASES[K]]["pipeline"]>}} AliasTasks A mapping from pipeline aliases to their corresponding pipeline classes.
 * @typedef {SupportedTasks & AliasTasks} AllTasks A mapping from all pipeline names and aliases to their corresponding pipeline classes.
 */

/**
 * Utility factory method to build a `Pipeline` object.
 *
 * @template {PipelineType} T The type of pipeline to return.
 * @param {T} task The task defining which pipeline will be returned. Currently accepted tasks are:
 *  - `"automatic-speech-recognition"`: will return a `AutomaticSpeechRecognitionPipeline`.
 *  - `"depth-estimation"`: will return a `DepthEstimationPipeline`.
 *  - `"document-question-answering"`: will return a `DocumentQuestionAnsweringPipeline`.
 *  - `"feature-extraction"`: will return a `FeatureExtractionPipeline`.
 *  - `"fill-mask"`: will return a `FillMaskPipeline`.
 *  - `"question-answering"`: will return a `QuestionAnsweringPipeline`.
 *  - `"summarization"`: will return a `SummarizationPipeline`.
 *  - `"text2text-generation"`: will return a `Text2TextGenerationPipeline`.
 *  - `"text-classification"` (alias "sentiment-analysis" available): will return a `TextClassificationPipeline`.
 *  - `"text-generation"`: will return a `TextGenerationPipeline`.
 *  - `"token-classification"` (alias "ner" available): will return a `TokenClassificationPipeline`.
 *  - `"translation"`: will return a `TranslationPipeline`.
 *  - `"translation_xx_to_yy"`: will return a `TranslationPipeline`.
 * @param {string} [model=null] The name of the pre-trained model to use. If not specified, the default model for the task will be used.
 * @param {import('./utils/hub.js').PretrainedOptions} [options] Optional parameters for the pipeline.
 * @returns {Promise<AllTasks[T]>} A Pipeline object for the specified task.
 * @throws {Error} If an unsupported pipeline is requested.
 */
export async function pipeline(
    task,
    model = null,
    {
        quantized = true,
        progress_callback = null,
        config = null,
        cache_dir = null,
        local_files_only = false,
        revision = 'main',
        model_file_name = null,
    } = {}
) {
    // Helper method to construct pipeline

    // Apply aliases
    // @ts-ignore
    task = TASK_ALIASES[task] ?? task;

    // Get pipeline info
    const pipelineInfo = SUPPORTED_TASKS[task.split('_', 1)[0]];
    if (!pipelineInfo) {
        throw Error(`Unsupported pipeline: ${task}. Must be one of [${Object.keys(SUPPORTED_TASKS)}]`)
    }

    // Use model if specified, otherwise, use default
    if (!model) {
        model = pipelineInfo.default.model
        console.log(`No model specified. Using default model: "${model}".`);
    }

    const pretrainedOptions = {
        quantized,
        progress_callback,
        config,
        cache_dir,
        local_files_only,
        revision,
        model_file_name,
    }

    const classes = new Map([
        ['tokenizer', pipelineInfo.tokenizer],
        ['model', pipelineInfo.model],
        ['processor', pipelineInfo.processor],
    ]);

    // Load model, tokenizer, and processor (if they exist)
    const results = await loadItems(classes, model, pretrainedOptions);
    results.task = task;

    dispatchCallback(progress_callback, {
        'status': 'ready',
        'task': task,
        'model': model,
    });

    const pipelineClass = pipelineInfo.pipeline;
    return new pipelineClass(results);
}


/**
 * Helper function to get applicable model, tokenizer, or processor classes for a given model.
 * @param {Map<string, any>} mapping The mapping of names to classes, arrays of classes, or null.
 * @param {string} model The name of the model to load.
 * @param {import('./utils/hub.js').PretrainedOptions} pretrainedOptions The options to pass to the `from_pretrained` method.
 * @private
 */
async function loadItems(mapping, model, pretrainedOptions) {

    const result = Object.create(null);

    /**@type {Promise[]} */
    const promises = [];
    for (let [name, cls] of mapping.entries()) {
        if (!cls) continue;

        /**@type {Promise} */
        let promise;
        if (Array.isArray(cls)) {
            promise = new Promise(async (resolve, reject) => {
                let e;
                for (let c of cls) {
                    if (c === null) {
                        // If null, we resolve it immediately, meaning the relevant
                        // class was not found, but it is optional.
                        resolve(null);
                        return;
                    }
                    try {
                        resolve(await c.from_pretrained(model, pretrainedOptions));
                        return;
                    } catch (err) {
                        e = err;
                    }
                }
                reject(e);
            })
        } else {
            promise = cls.from_pretrained(model, pretrainedOptions);
        }

        result[name] = promise;
        promises.push(promise);
    }

    // Wait for all promises to resolve (in parallel)
    await Promise.all(promises);

    // Then assign to result
    for (let [name, promise] of Object.entries(result)) {
        result[name] = await promise;
    }

    return result;
}
