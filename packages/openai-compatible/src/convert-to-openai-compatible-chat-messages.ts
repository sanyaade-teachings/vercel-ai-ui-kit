import {
  LanguageModelV1Message,
  LanguageModelV1Prompt,
  UnsupportedFunctionalityError,
} from '@ai-sdk/provider';
import { convertUint8ArrayToBase64 } from '@ai-sdk/provider-utils';
import { OpenAICompatibleChatPrompt } from './openai-compatible-api-types';

function findAndHoistMetadata(obj: LanguageModelV1Message) {
  if (obj?.providerMetadata?.openaiCompatible) {
    Object.assign(obj, obj.providerMetadata.openaiCompatible);
    delete obj.providerMetadata.openaiCompatible;
    if (Object.keys(obj.providerMetadata).length === 0) {
      delete obj.providerMetadata;
    }
  }

  if (Array.isArray(obj)) {
    obj.forEach(item => findAndHoistMetadata(item));
  } else if (obj && typeof obj === 'object') {
    Object.values(obj).forEach(value => findAndHoistMetadata(value));
  }
}

export function convertToOpenAICompatibleChatMessages(
  prompt: LanguageModelV1Prompt,
): OpenAICompatibleChatPrompt {
  const messages: OpenAICompatibleChatPrompt = [];

  console.log(`Original prompt: ${JSON.stringify(prompt, null, 2)}`);
  // let promptCopy = { ...prompt };
  for (let i = 0; i < prompt.length; i++) {
    findAndHoistMetadata(prompt[i]);
  }
  console.log(`Prompt after hoisting: ${JSON.stringify(prompt, null, 2)}`);

  for (const { role, content, ...rest } of prompt) {

    console.log(`Role: ${role}`);
    console.log(`Content: ${JSON.stringify(content, null, 2)}`);
    console.log(`Rest: ${JSON.stringify(rest, null, 2)}`);

    switch (role) {
      case 'system': {
        messages.push({ role: 'system', content, ...rest });
        break;
      }

      case 'user': {
        if (content.length === 1 && content[0].type === 'text') {
          console.log('Single text part');
          const { text, type, ...contentRest } = content[0];
          messages.push({ role: 'user', content: text, ...rest, ...contentRest });
          break;
        }

        messages.push({
          role: 'user',
          content: content.map(part => {
            // Intentionally throw away `...partRest` here as we handle including
            // any remaining material within each case individually below.
            const { type, ...partRest } = part;
            switch (type) {
              case 'text': {
                const { type, text, ...textRest } = part;
                return { type, text, ...textRest };
              }
              case 'image': {
                const { type, image, mimeType, ...imageRest } = part;
                return {
                  type: 'image_url',
                  image_url: {
                    url:
                      image instanceof URL
                        ? image.toString()
                        : `data:${mimeType ?? 'image/jpeg'
                        };base64,${convertUint8ArrayToBase64(image)}`,
                  },
                  ...imageRest,
                };
              }
              case 'file': {
                throw new UnsupportedFunctionalityError({
                  functionality: 'File content parts in user messages',
                });
              }
            }
          }),
          ...rest,
        });

        break;
      }

      case 'assistant': {
        let text = '';
        const toolCalls: Array<{
          id: string;
          type: 'function';
          function: { name: string; arguments: string };
        }> = [];

        for (const part of content) {
          switch (part.type) {
            case 'text': {
              // TODO: Check whether there are valid test cases here where we
              // could lose information by not including `...rest`.
              text += part.text;
              break;
            }
            case 'tool-call': {
              const { type, toolCallId, toolName, args, ...toolCallRest } = part;
              toolCalls.push({
                id: toolCallId,
                type: 'function',
                function: {
                  name: toolName,
                  arguments: JSON.stringify(args),
                },
                ...toolCallRest,
              });
              break;
            }
            default: {
              const _exhaustiveCheck: never = part;
              throw new Error(`Unsupported part: ${_exhaustiveCheck}`);
            }
          }
        }

        messages.push({
          role: 'assistant',
          content: text,
          tool_calls: toolCalls.length > 0 ? toolCalls : undefined,
          ...rest,
        });

        break;
      }

      case 'tool': {
        for (const toolResponse of content) {
          const { type, content, toolCallId, toolName, result, ...toolResponseRest } = toolResponse;
          messages.push({
            role: 'tool',
            tool_call_id: toolCallId,
            content: JSON.stringify(result),
            ...toolResponseRest,
            ...rest,
          });
        }
        break;
      }

      default: {
        const _exhaustiveCheck: never = role;
        throw new Error(`Unsupported role: ${_exhaustiveCheck}`);
      }
    }
  }

  return messages;
}
