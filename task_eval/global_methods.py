import openai
import numpy as np
import json
import time
import sys
import os

import google.generativeai as genai
from anthropic import Anthropic
import requests


def get_openai_embedding(texts, model="text-embedding-ada-002"):
    texts = [text.replace("\n", " ") for text in texts]
    return np.array(
        [
            openai.Embedding.create(input=texts, model=model)["data"][i]["embedding"]
            for i in range(len(texts))
        ]
    )


def set_anthropic_key():
    pass


def set_gemini_key():
    # Or use `os.getenv('GOOGLE_API_KEY')` to fetch an environment variable.
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])


def set_openai_key():
    openai.api_key = os.environ["OPENAI_API_KEY"]


def run_json_trials(
    query,
    num_gen=1,
    num_tokens_request=1000,
    model="davinci",
    use_16k=False,
    temperature=1.0,
    wait_time=1,
    examples=None,
    input=None,
):
    run_loop = True
    counter = 0
    while run_loop:
        try:
            if examples is not None and input is not None:
                output = run_chatgpt_with_examples(
                    query,
                    examples,
                    input,
                    num_gen=num_gen,
                    wait_time=wait_time,
                    num_tokens_request=num_tokens_request,
                    use_16k=use_16k,
                    temperature=temperature,
                ).strip()
            else:
                output = run_chatgpt(
                    query,
                    num_gen=num_gen,
                    wait_time=wait_time,
                    model=model,
                    num_tokens_request=num_tokens_request,
                    use_16k=use_16k,
                    temperature=temperature,
                )
            output = output.replace("json", "")  # this frequently happens
            facts = json.loads(output.strip())
            run_loop = False
        except json.decoder.JSONDecodeError:
            counter += 1
            time.sleep(1)
            print("Retrying to avoid JsonDecodeError, trial %s ..." % counter)
            print(output)
            if counter == 10:
                print("Exiting after 10 trials")
                sys.exit()
            continue
    return facts


def run_claude(query, max_new_tokens, model_name):
    if model_name == "claude-sonnet":
        model_name = "claude-3-sonnet-20240229"
    elif model_name == "claude-haiku":
        model_name = "claude-3-haiku-20240307"

    client = Anthropic(
        # This is the default and can be omitted
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )
    # print(query)
    message = client.messages.create(
        max_tokens=max_new_tokens,
        messages=[
            {
                "role": "user",
                "content": query,
            }
        ],
        model=model_name,
    )
    print(message.content)
    return message.content[0].text


def run_llama_open_router(
    query,
    max_new_tokens,
    *,
    model="thinkingmachines/inkling:free",
    api_key=None,
    max_retries=5,
):
    """OpenRouter chat-completions adapter. Same call shape as run_llama_groq
    (query, max_new_tokens, model=, api_key=). Select it with --provider
    openrouter; Groq remains available when rate limits require it.
    """
    openrouter_api_key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY is not set.")

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": query}],
        "max_tokens": max_new_tokens,
    }
    headers = {
        "Authorization": f"Bearer {openrouter_api_key}",
        "Content-Type": "application/json",
    }

    for attempt in range(max_retries):
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=300,
        )
        if response.status_code == 429:
            wait = int(response.headers.get("Retry-After", 5 * (attempt + 1)))
            print(
                f"Rate limited (429). Waiting {wait}s (attempt {attempt+1}/{max_retries})..."
            )
            time.sleep(wait)
            continue

        if not response.ok:
            print(f"OpenRouter error {response.status_code}: {response.text}")
        response.raise_for_status()
        data = response.json()
        message = data["choices"][0]["message"]
        content = message.get("content")
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        if not content:
            raise RuntimeError(f"Empty OpenRouter content: {data}")
        return content

    raise RuntimeError(f"Exceeded {max_retries} retries due to rate limiting.")


def run_llama_groq(
    query,
    max_new_tokens,
    *,
    model="llama-3.3-70b-versatile",
    api_key=None,
    max_retries=5,
):
    """Groq equivalent of run_llama_open_router. Same call shape (query,
    max_new_tokens, model=, api_key=) so it's a drop-in swap at call sites.
    Uses Groq's OpenAI-compatible REST endpoint directly rather than the
    groq SDK, so the same requests-based error handling (status codes,
    Retry-After, raw error bodies) applies here too.

    `model` must be a Groq model slug (e.g. "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant") -- NOT an OpenRouter slug like
    "meta-llama/llama-3.3-70b-instruct:free". Pass the right one via
    --model on the command line.
    """
    groq_api_key = api_key or os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY is not set.")

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": query}],
        "max_tokens": max_new_tokens,
    }
    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json",
    }

    for attempt in range(max_retries):
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=300,
        )
        if response.status_code == 429:
            wait = int(response.headers.get("Retry-After", 5 * (attempt + 1)))
            print(
                f"Rate limited (429) on Groq. Waiting {wait}s (attempt {attempt+1}/{max_retries})..."
            )
            time.sleep(wait)
            continue

        if not response.ok:
            print(f"Groq error {response.status_code}: {response.text}")
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    raise RuntimeError(f"Exceeded {max_retries} retries due to rate limiting (Groq).")


def resolve_llm(provider="openrouter"):
    """Return the existing OpenRouter or Groq function. Does not wrap either."""
    if provider == "openrouter":
        return run_llama_open_router
    if provider == "groq":
        return run_llama_groq
    raise ValueError(
        f"Unknown LLM provider {provider!r}; use 'openrouter' or 'groq'."
    )


def run_gemini(model, content: str, max_tokens: int = 0):
    try:
        response = model.generate_content(content)
        return response.text
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
        return None


def run_chatgpt(
    query,
    num_gen=1,
    num_tokens_request=1000,
    model="chatgpt",
    use_16k=False,
    temperature=1.0,
    wait_time=1,
):
    completion = None
    while completion is None:
        wait_time = wait_time * 2
        try:
            # if model == 'davinci':
            #     completion = openai.Completion.create(
            #                     # model = "gpt-3.5-turbo",
            #                     model = "text-davinci-003",
            #                     temperature = temperature,
            #                     max_tokens = num_tokens_request,
            #                     n=num_gen,
            #                     prompt=query
            #                 )
            if model == "chatgpt":
                messages = [{"role": "system", "content": query}]
                completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    temperature=temperature,
                    max_tokens=num_tokens_request,
                    n=num_gen,
                    messages=messages,
                )
            elif "gpt-4" in model:
                completion = openai.ChatCompletion.create(
                    model=model,
                    temperature=temperature,
                    max_tokens=num_tokens_request,
                    n=num_gen,
                    messages=[{"role": "user", "content": query}],
                )
            else:
                print("Did not find model %s" % model)
                raise ValueError
        except openai.error.APIError as e:
            # Handle API error here, e.g. retry or log
            print(
                f"OpenAI API returned an API Error: {e}; waiting for {wait_time} seconds"
            )
            time.sleep(wait_time)
            pass
        except openai.error.APIConnectionError as e:
            # Handle connection error here
            print(
                f"Failed to connect to OpenAI API: {e}; waiting for {wait_time} seconds"
            )
            time.sleep(wait_time)
            pass
        except openai.error.RateLimitError as e:
            # Handle rate limit error (we recommend using exponential backoff)
            print(f"OpenAI API request exceeded rate limit: {e}")
            pass
        except openai.error.ServiceUnavailableError as e:
            # Handle rate limit error (we recommend using exponential backoff)
            print(
                f"OpenAI API request exceeded rate limit: {e}; waiting for {wait_time} seconds"
            )
            time.sleep(wait_time)
            pass
        # except Exception as e:
        #     if e:
        #         print(e)
        #         print(f"Timeout error, retrying after waiting for {wait_time} seconds")
        #         time.sleep(wait_time)

    if model == "davinci":
        outputs = [choice.get("text").strip() for choice in completion.get("choices")]
        if num_gen > 1:
            return outputs
        else:
            # print(outputs[0])
            return outputs[0]
    else:
        # print(completion.choices[0].message.content)
        return completion.choices[0].message.content


def run_chatgpt_with_examples(
    query,
    examples,
    input,
    num_gen=1,
    num_tokens_request=1000,
    use_16k=False,
    wait_time=1,
    temperature=1.0,
):
    completion = None

    messages = [{"role": "system", "content": query}]
    for inp, out in examples:
        messages.append({"role": "user", "content": inp})
        messages.append({"role": "system", "content": out})
    messages.append({"role": "user", "content": input})

    while completion is None:
        wait_time = wait_time * 2
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo" if not use_16k else "gpt-3.5-turbo-16k",
                temperature=temperature,
                max_tokens=num_tokens_request,
                n=num_gen,
                messages=messages,
            )
        except openai.error.APIError as e:
            # Handle API error here, e.g. retry or log
            print(
                f"OpenAI API returned an API Error: {e}; waiting for {wait_time} seconds"
            )
            time.sleep(wait_time)
            pass
        except openai.error.APIConnectionError as e:
            # Handle connection error here
            print(
                f"Failed to connect to OpenAI API: {e}; waiting for {wait_time} seconds"
            )
            time.sleep(wait_time)
            pass
        except openai.error.RateLimitError as e:
            # Handle rate limit error (we recommend using exponential backoff)
            print(f"OpenAI API request exceeded rate limit: {e}")
            pass
        except openai.error.ServiceUnavailableError as e:
            # Handle rate limit error (we recommend using exponential backoff)
            print(
                f"OpenAI API request exceeded rate limit: {e}; waiting for {wait_time} seconds"
            )
            time.sleep(wait_time)
            pass

    return completion.choices[0].message.content
