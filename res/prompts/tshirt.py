"""This module contains prompts for the application."""

# General Product Description for blueprint 6
blueprint_6_description = """
General Product Description:
Unisex Gildan T-Shirt - Made with medium fabric (5.3 oz/yd² (180 g/m²)) consisting of 100% cotton for year-round comfort that is sustainable and highly durable. The classic fit of this shirt ensures a comfy, relaxed wear while the crew neckline adds that neat, timeless look that can blend into any occasion, casual or semi-formal.
"""

# Example response format
example_format = """
{
  "patterns": [
    {
      "product_name": "Funny Teacher Math Equation",
      "description": "Add some humor to your classroom wardrobe with this witty teacher-themed t-shirt. Made from soft, durable cotton, it's perfect for those long days of shaping young minds.",
      "tshirt_text": "1 Teacher + 25 Students = Endless Possibilities",
      "marketing_tags": ["teacher humor", "education", "classroom style", "math teacher", "school life"]
    }
  ]
}
"""

# AI prompt
user_message = f"""Generate unique t-shirt designs for this idea: %s. I need %s designs.

Response must be in this exact JSON format:
{example_format}

Each design should have:
1. product_name: Catchy, SEO-friendly product title
2. description: Engaging product description that mentions both the design and the shirt quality
3. tshirt_text: The actual text to be printed on the shirt (be creative and unique!)
4. marketing_tags: 5-7 relevant tags for marketing

The t-shirts are:
{blueprint_6_description}

Make the designs witty, creative, and appealing to your target audience. Focus on:
- Clear, memorable text
- Professional yet engaging tone
- Unique ideas that stand out
- Marketable concepts

Remember to format the response as valid JSON!
"""
