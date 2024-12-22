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
  prompt.map(message => findAndHoistMetadata(message));
  for (const { role, content, ...rest } of prompt) {
    switch (role) {
      case 'system': {
        messages.push({ role: 'system', content, ...rest });
        break;
      }

      case 'user': {
        if (content.length === 1 && content[0].type === 'text') {
          const { text, type, ...contentRest } = content[0];
          messages.push({ role: 'user', content: text, ...rest, ...contentRest });
          break;
        }

        messages.push({
          role: 'user',
          content: content.map(part => {
            switch (part.type) {
              case 'text': {
                return { ...part };
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
              // We could be throwing away additional data here as we only
              // incorporate `part.text`. However, it's not clear there are use
              // cases requiring this, nor how we'd resolve/merge across
              // potentially multiple text parts.
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
