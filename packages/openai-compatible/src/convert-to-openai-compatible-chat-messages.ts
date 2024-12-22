import {
  LanguageModelV1Prompt,
  LanguageModelV1ProviderMetadata,
  UnsupportedFunctionalityError,
} from '@ai-sdk/provider';
import { convertUint8ArrayToBase64 } from '@ai-sdk/provider-utils';
import { OpenAICompatibleChatPrompt } from './openai-compatible-api-types';

function getOpenAIMetadata(message: { providerMetadata?: LanguageModelV1ProviderMetadata }) {
  return message?.providerMetadata?.openaiCompatible ?? {};
}

export function convertToOpenAICompatibleChatMessages(
  prompt: LanguageModelV1Prompt,
): OpenAICompatibleChatPrompt {
  const messages: OpenAICompatibleChatPrompt = [];
  for (const message of prompt) {
    const metadata = getOpenAIMetadata(message);
    const { role, content } = message;
    switch (role) {
      case 'system': {
        messages.push({ role: 'system', content, ...metadata });
        break;
      }

      case 'user': {
        if (content.length === 1 && content[0].type === 'text') {
          messages.push({ role: 'user', content: content[0].text, ...getOpenAIMetadata(content[0]) });
          break;
        }

        messages.push({
          role: 'user',
          content: content.map(part => {
            const partMetadata = getOpenAIMetadata(part);
            switch (part.type) {
              case 'text': {
                return { type: 'text', text: part.text, ...partMetadata };
              }
              case 'image': {
                const { image, mimeType } = part;
                return {
                  type: 'image_url',
                  image_url: {
                    url:
                      image instanceof URL
                        ? image.toString()
                        : `data:${mimeType ?? 'image/jpeg'
                        };base64,${convertUint8ArrayToBase64(image)}`,
                  },
                  ...partMetadata,
                };
              }
              case 'file': {
                throw new UnsupportedFunctionalityError({
                  functionality: 'File content parts in user messages',
                });
              }
            }
          }),
          ...metadata,
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
          const partMetadata = getOpenAIMetadata(part);
          switch (part.type) {
            case 'text': {
              // We could be throwing away additional data here as we only
              // incorporate `part.text`. However, use cases aren't clear, nor
              // how we'd resolve/merge across potentially multiple parts.
              text += part.text;
              break;
            }
            case 'tool-call': {
              const { toolCallId, toolName, args } = part;
              toolCalls.push({
                id: toolCallId,
                type: 'function',
                function: {
                  name: toolName,
                  arguments: JSON.stringify(args),
                },
                ...partMetadata,
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
          ...metadata,
        });

        break;
      }

      case 'tool': {
        for (const toolResponse of content) {
          const { toolCallId, result } = toolResponse;
          const toolResponseMetadata = getOpenAIMetadata(toolResponse);
          messages.push({
            role: 'tool',
            tool_call_id: toolCallId,
            content: JSON.stringify(result),
            ...toolResponseMetadata,
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
