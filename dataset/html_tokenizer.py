import re

class HTMLTailwindPreprocessor:
    """
    Stage 1 of the hybrid tokenizer: Pre-segments raw HTML+Tailwind CSS code
    into a list of meaningful "proto-tokens".
    These proto-tokens will then be fed into a BPE (Byte Pair Encoding) model.
    """

    def _tokenize_attributes(self, attributes_str: str) -> list[str]:
        """
        Helper function to tokenize a string of HTML attributes.
        Example: 'class="foo bar" id="baz" required'
        Becomes: ['class="', 'foo', 'bar', '"', 'id="', 'baz', '"', 'required']
        """
        tokens = []
        # Regex to find individual attributes: name="value", name='value', name=value_unquoted, or just name (boolean)
        # Group 1: attribute name
        # Group 2: ="value" (double quoted)
        # Group 3: ='value' (single quoted)
        # Group 4: =value_unquoted
        # We primarily care about the name and the raw value string (if any)
        attr_regex = re.compile(
            r'([a-zA-Z0-9_:-]+)'  # Attribute name
            r'(?:\s*=\s*(?:'  # Optional value part
            r'"([^"]*)"|'  # Double-quoted value (Group 2)
            r"'([^']*)'|"  # Single-quoted value (Group 3)
            r'([^>\s\'"]+)'  # Unquoted value (Group 4)
            r'))?'
        )
        current_pos = 0
        while current_pos < len(attributes_str):
            match = attr_regex.match(attributes_str, current_pos)
            if not match:
                current_pos += 1 # Skip character if no match (e.g. extra spaces)
                continue

            attr_name = match.group(1)
            
            # Check if it's an attribute with a value
            # Value could be in group 2, 3, or 4
            value_double_quoted = match.group(2)
            value_single_quoted = match.group(3)
            value_unquoted = match.group(4)

            if value_double_quoted is not None or \
               value_single_quoted is not None or \
               value_unquoted is not None:
                
                tokens.append(f'{attr_name}="') # Normalize to class="

                actual_value = value_double_quoted if value_double_quoted is not None else \
                               value_single_quoted if value_single_quoted is not None else \
                               value_unquoted

                if actual_value is not None: # Should always be true if one of the value groups matched
                    if attr_name.lower() == "class":
                        # Split Tailwind CSS classes by space
                        for cls_token in actual_value.split():
                            if cls_token: # Ensure non-empty
                                tokens.append(cls_token)
                    else:
                        # For other attributes, keep the value as a single proto-token.
                        # BPE will handle further sub-tokenization if needed.
                        tokens.append(actual_value)
                
                tokens.append('"') # Closing quote for the attribute value token
            else:
                # Boolean attribute (e.g., "required", "disabled")
                tokens.append(attr_name)
            
            current_pos = match.end()
            # Skip any spaces before the next attribute
            while current_pos < len(attributes_str) and attributes_str[current_pos].isspace():
                current_pos +=1
        return tokens

    def pre_segment(self, html_string: str) -> list[str]:
        """
        Performs HTML-aware pre-segmentation.
        Converts an HTML string into a list of proto-tokens.
        """
        proto_tokens = []
        
        # 1. Normalize multiple whitespaces to a single space, and trim ends.
        # This simplifies subsequent regex.
        processed_html = re.sub(r'\\s+', ' ', html_string).strip()

        # 2. Split the HTML string by tags, keeping the tags as delimiters.
        # re.split with a capturing group (the parentheses around <[^>]+>)
        # results in a list where delimiters are also included.
        # Example: "text <tag1> text2 </tag2>" -> ["text ", "<tag1>", " text2 ", "</tag2>", ""]
        parts = re.split(r'(<!--.*?-->|<!DOCTYPE[^>]*>|<[^>]+>)', processed_html)

        for part in parts:
            stripped_part = part.strip() # Clean whitespace for each part
            if not stripped_part:
                continue

            # Check if the part is a comment, DOCTYPE, or a general tag
            if stripped_part.startswith("<!--") and stripped_part.endswith("-->"):
                proto_tokens.append(stripped_part) # HTML Comment
            elif stripped_part.startswith("<!DOCTYPE"):
                proto_tokens.append(stripped_part) # DOCTYPE declaration
            elif stripped_part.startswith("<") and stripped_part.endswith(">"):
                # This is a tag, e.g., "<div class=\"foo\">", "</div>"
                
                # Handle closing tags first: e.g., </div>, </p>
                if stripped_part.startswith("</"):
                    proto_tokens.append(stripped_part) # Full closing tag as one token
                    continue

                # Process opening or self-closing tags
                # Example: <div class="foo bar" id="baz"> or <img src="pic.jpg"/>
                tag_content_full = stripped_part[1:-1].strip() # Content between < and >

                is_self_closing = tag_content_full.endswith('/')
                if is_self_closing:
                    tag_content_full = tag_content_full[:-1].strip() # Remove trailing /

                # Split the tag name from its attributes string
                # E.g., "div class=\"foo bar\" id=\"baz\"" -> name="div", attrs="class=\"foo bar\" id=\"baz\""
                tag_name_parts = tag_content_full.split(maxsplit=1)
                tag_name = tag_name_parts[0]
                
                proto_tokens.append(f"<{tag_name}") # Token: <div, <img

                attributes_str = ""
                if len(tag_name_parts) > 1:
                    attributes_str = tag_name_parts[1].strip()
                
                if attributes_str:
                    proto_tokens.extend(self._tokenize_attributes(attributes_str))
                
                if is_self_closing:
                    proto_tokens.append("/>") # Token: />
                else:
                    proto_tokens.append(">") # Token: >
            else:
                # This part is text content between tags
                # The user specification implies "Text content" can be a single proto-token.
                # BPE will handle further breaking it down if needed.
                # If you prefer to split text content by spaces here, you can uncomment the next lines:
                for word in stripped_part.split():
                   if word: proto_tokens.append(word)
                # proto_tokens.append(stripped_part) # Text content as one token
        
        return [token for token in proto_tokens if token] # Final cleanup for any empty strings

# Example Usage (for testing this file directly)
if __name__ == '__main__':
    preprocessor = HTMLTailwindPreprocessor()
    
    print("--- Test Case 1 ---")
    # html1 = '<div class="text-blue-500 font-bold">Hello World</div>'
    with open("dataset/small-dataset/image_0.html", "r") as f:
        html1 = f.read()
    tokens1 = preprocessor.pre_segment(html1)
    # Expected: ['<div', 'class="', 'text-blue-500', 'font-bold', '"', '>', 'Hello World', '</div>']
    print(f"Original: {html1}")
    print(f"Tokens: {tokens1}\\n")

    # print("--- Test Case 2 ---")
    # html2 = '<img src="image.jpg" alt="An image" required /> <!-- comment here -->'
    # tokens2 = preprocessor.pre_segment(html2)
    # # Expected: ['<img', 'src="', 'image.jpg', '"', 'alt="', 'An image', '"', 'required', '/>', '<!-- comment here -->']
    # print(f"Original: {html2}")
    # print(f"Tokens: {tokens2}\\n")

    # print("--- Test Case 3 ---")
    # html3 = "<!DOCTYPE html> <p data-value='test value' class='foo\\tbar baz'> Text </p>"
    # tokens3 = preprocessor.pre_segment(html3)
    # # Expected: ['<!DOCTYPE html>', '<p', 'data-value="', 'test value', '"', 'class="', 'foo', 'bar', 'baz', '"', '>', 'Text', '</p>']
    # print(f"Original: {html3}")
    # print(f"Tokens: {tokens3}\\n")
    
    # print("--- Test Case 4 ---")
    # html4 = "No tags here, just text."
    # tokens4 = preprocessor.pre_segment(html4)
    # # Expected: ["No tags here, just text."]
    # print(f"Original: {html4}")
    # print(f"Tokens: {tokens4}\\n")

    # print("--- Test Case 5 ---")
    # html5 = "<br>" # Simple tag with no attributes
    # tokens5 = preprocessor.pre_segment(html5)
    # # Expected: ["<br", ">"]
    # print(f"Original: {html5}")
    # print(f"Tokens: {tokens5}\\n")
    
    # print("--- Test Case 6 ---")
    # html6 = "<custom-tag attribute = value-no-quote >content</custom-tag>"
    # tokens6 = preprocessor.pre_segment(html6)
    # # Expected: ["<custom-tag", "attribute="", "value-no-quote", """, ">", "content", "</custom-tag>"]
    # print(f"Original: {html6}")
    # print(f"Tokens: {tokens6}\\n")

    # print("--- Test Case 7 ---")
    # html7 = "<div class=\"\">Empty class</div>" # Empty class attribute
    # tokens7 = preprocessor.pre_segment(html7)
    # # Expected: ['<div', 'class="', '"', '>', 'Empty class', '</div>']
    # print(f"Original: {html7}")
    # print(f"Tokens: {tokens7}\\n")
    
    # print("--- Test Case 8 ---")
    # html8 = "<div class = \"  m-4  p-2 \" > Spaced attributes </div>"
    # tokens8 = preprocessor.pre_segment(html8)
    # # Expected: ['<div', 'class="', 'm-4', 'p-2', '"', '>', 'Spaced attributes', '</div>']
    # print(f"Original: {html8}")
    # print(f"Tokens: {tokens8}\\n") 