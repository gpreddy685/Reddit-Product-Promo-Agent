import streamlit as st
import os
from groq import Groq
from dotenv import load_dotenv
import praw 
from typing import List, Dict, Any, Optional
from langchain_groq import ChatGroq
from langchain import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import Tool
from langchain.agents import create_react_agent, AgentExecutor
import time
import re

st.set_page_config(
    page_title="Reddit Reply Agent",
    page_icon="💬",
    layout="wide"
)

load_dotenv()

if "GROQ_API_KEY" not in st.session_state:
    st.session_state.GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

def initialize_reddit(client_id, client_secret, user_agent, username, password):
    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        username=username,
        password=password,
    )
    return reddit

def extract_subreddits_with_groq(user_input: str, api_key: str) -> List[str]:
    """Extract potential subreddit names from user input using Groq LLM."""
    groq_client = Groq(api_key=api_key)
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Extract at most 10 keywords that can possibly be subreddit names based on: {user_input}. Just return the keywords separated by commas. No prefix or suffix needed."},
        ],
        model="llama-3.3-70b-versatile",  
    )

    suggested_subreddits = chat_completion.choices[0].message.content
    suggested_subreddits = suggested_subreddits.replace("\n", ",")  
    suggested_subreddits = [sub.strip() for sub in suggested_subreddits.split(",") if sub.strip()]  # Clean up spaces

    return suggested_subreddits

def fetch_relevant_posts(reddit: praw.Reddit, subreddits: List[str], keyword: str) -> List[Dict[str, Any]]:
    """Fetch relevant posts from valid subreddits."""
    relevant_posts = []

    for subreddit in subreddits:
        try:
            subreddit_obj = reddit.subreddit(subreddit)
            
            for post in subreddit_obj.search(query=keyword, limit=10):
                post_data = {
                    "post_id": post.id,
                    "title": post.title,
                    "url": post.url,
                    "subreddit": subreddit,
                    "num_comments": post.num_comments,
                    "score": post.score,
                    "selftext": post.selftext if hasattr(post, 'selftext') else ""
                }
                relevant_posts.append(post_data)

        except Exception as e:
            continue

    return relevant_posts
    
def filter_posts_with_llm(user_input: str, keyword: str, posts: List[Dict[str, Any]], api_key: str) -> List[Dict[str, Any]]:
    """Use LLM to filter out irrelevant posts based on user interest and keyword."""
    filtered_posts = []
    groq_client = Groq(api_key=api_key)

    for post in posts:
        llm_prompt = f"""
        You are an AI that filters relevant Reddit posts for a user.
        
        **User's Input (Product/Service & Goal):** 
        {user_input}

        **Keyword:** 
        {keyword}

        **Reddit Post Title:** 
        {post['title']}

        The user is looking for posts that:
        - Mention their product/service directly
        - Discuss problems related to their domain (even if their product is not mentioned)
        - Talk about solutions, communities, or technologies related to their work

        **Question:**  
        Does this post seem relevant to the user's interest based on the above criteria?  
        Answer only with 'YES' or 'NO'.
        """

        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an AI that determines if a Reddit post is relevant to the user's product, service, or problem domain."},
                {"role": "user", "content": llm_prompt},
            ],
            model="llama-3.3-70b-versatile",  
        )

        response = chat_completion.choices[0].message.content.strip().upper()
        
        if response == "YES":
            filtered_posts.append(post)

    return filtered_posts

def generate_comment(user_input: str, post_title: str, post_content: str, api_key: str) -> str:
    """Generates a natural, casual Reddit comment with a slight Hinglish touch."""
    groq_client = Groq(api_key=api_key)
    prompt = f"""
    You are a regular Reddit user replying to a post. Your tone should be **casual, friendly, and chill**, like a normal human conversation.

    The Reddit post is titled: {post_title}
    The post content is: {post_content}

    Based on this, write a response that:
    - Feels **natural & informal**, not overly scripted.
    - Is mostly **English** with a **tiny touch** of Hinglish (just a word or two, like "bro," "scene," "thoda," etc.).
    - Is **not motivational**—keep it real.
    - Stays **relevant to stuttering** without sounding preachy.
    - If relevant, mentions **StutterEase casually**, without pushing it.

    Example tone:
    - "Dude, I totally get this. Talking to people can be a scene sometimes. But lowkey, one thing that helped me was..."
    - "Yeah bro, same here. Practicing in front of a mirror sounded dumb at first, but it actually worked lol."
    - "Oh man, I feel you. Have you tried those AI convo apps? Kinda cool for low-pressure practice."

    Now generate a response in a **similar tone**.
    """

    chat_completion = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a casual Reddit user. Your tone is friendly, chill, and mostly English with a slight Hinglish touch."},
            {"role": "user", "content": prompt},
        ],
        model="llama-3.3-70b-versatile",  
    )

    return chat_completion.choices[0].message.content.strip()

# Custom callback handler to stream thoughts
class StreamingCallbackHandler:
    def __init__(self, container):
        self.container = container
        self.thought_text = ""
        self.messages = []
    
    def on_agent_action(self, action, run_id=None):
        # Extract tool, tool input
        tool = action.tool
        tool_input = action.tool_input
        
        thought = f"🤔 **Thinking:** I'll use the {tool} tool with input: {tool_input}\n\n"
        self.thought_text += thought
        self.container.markdown(self.thought_text)
        
    def on_agent_finish(self, finish, run_id=None):
        thought = f"✅ **Finished:** {finish.return_values['output']}\n\n"
        self.thought_text += thought
        self.container.markdown(self.thought_text)

    def on_llm_new_token(self, token, run_id=None):
        self.thought_text += token
        self.container.markdown(self.thought_text)
    
    def on_llm_start(self, serialized, prompts, run_id=None):
        thought = f"🧠 **Starting to think...**\n\n"
        self.thought_text += thought
        self.container.markdown(self.thought_text)
    
    def on_tool_start(self, serialized, input_str, run_id=None):
        thought = f"🔧 **Using tool...**\n\n"
        self.thought_text += thought
        self.container.markdown(self.thought_text)
    
    def on_tool_end(self, output, run_id=None):
        # Clean output for display
        clean_output = str(output)
        
        # If output is a list of dicts (like posts), format it nicely
        if clean_output.startswith('[') and ']' in clean_output and 'post_id' in clean_output:
            try:
                import ast
                posts = ast.literal_eval(clean_output)
                formatted_output = ""
                for i, post in enumerate(posts):
                    formatted_output += f"Post {i+1}: {post.get('title', 'No title')} in r/{post.get('subreddit', 'unknown')}\n"
                clean_output = formatted_output
            except:
                # If parsing fails, just use the original output
                pass
        
        thought = f"📋 **Tool result:**\n{clean_output}\n\n"
        self.thought_text += thought
        self.container.markdown(self.thought_text)

def setup_agent(api_key, reddit_instance, callback_handler=None):
    # Function to find relevant posts
    def get_relevant_reddit_posts(user_input: str, keyword: str) -> List[Dict[str, Any]]:
        suggested_subreddits = extract_subreddits_with_groq(user_input, api_key)
        all_posts = fetch_relevant_posts(reddit_instance, suggested_subreddits, keyword)
        filtered_posts = filter_posts_with_llm(user_input, keyword, all_posts, api_key)
        return filtered_posts

    # Function to post a comment - ACTUALLY POSTS TO REDDIT
    def post_reddit_comment(post_id: str, user_product_info: str, comment_seed: Optional[str] = None) -> str:
        try:
            post = reddit_instance.submission(id=post_id)
            
            comment_text = generate_comment(
                user_input=user_product_info,
                post_title=post.title,
                post_content=post.selftext if hasattr(post, 'selftext') and post.selftext else "No additional text available.",
                api_key=api_key
            )
            
            post.reply(comment_text)
            
            return f"✅ Comment successfully posted to Reddit on post: {post_id}\nComment: {comment_text}"
        except Exception as e:
            return f"❌ Failed to post comment on post {post_id}: {str(e)}"

    # Wrapper functions to handle string inputs
    def get_posts_wrapper(tool_input: str) -> List[Dict[str, Any]]:
        if "," in tool_input:
            parts = tool_input.split(",", 1)
            user_input = parts[0].strip()
            keyword = parts[1].strip()
        else:
            user_input = tool_input
            keyword = tool_input
        
        return get_relevant_reddit_posts(user_input, keyword)

    def post_comment_wrapper(tool_input: str) -> str:
        parts = tool_input.split(",")
        
        if len(parts) >= 2:
            post_id = parts[0].strip()
            user_product_info = parts[1].strip()
            comment_seed = None
            if len(parts) >= 3:
                comment_seed = parts[2].strip()
            
            return post_reddit_comment(post_id, user_product_info, comment_seed)
        else:
            return "Error: Not enough arguments. Please provide post_id and user_product_info separated by commas."

    get_posts_tool = Tool(
        name='GetRedditPosts',
        func=get_posts_wrapper,
        description='Find relevant Reddit posts. Input should be "user_input, keyword" where user_input is description of product/service and keyword is the primary search term.'
    )

    post_comment_tool = Tool(
        name='PostRedditComment',
        func=post_comment_wrapper,
        description='Generate and post a comment on a Reddit post. Input should be "post_id, user_product_info, [comment_seed]" where post_id is the Reddit post ID, user_product_info is information about product/service, and comment_seed is optional context.'
    )

    tools = [get_posts_tool, post_comment_tool]

    llm = ChatGroq(
        temperature=0.7,
        model="llama-3.3-70b-versatile",  
        api_key=api_key,
        streaming=True,
        callbacks=[callback_handler] if callback_handler else None
    )

    react_template = """Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought:{agent_scratchpad}"""

    prompt = PromptTemplate(
        template=react_template,
        input_variables=["tools", "tool_names", "input", "agent_scratchpad"]
    )

    memory = ConversationBufferWindowMemory(
        memory_key='chat_history',
        k=3,
        return_messages=True
    )

    reddit_agent = create_react_agent(
        tools=tools,
        llm=llm,
        prompt=prompt
    )

    agent_executor = AgentExecutor(
        agent=reddit_agent, 
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        callbacks=[callback_handler] if callback_handler else None
    )

    return agent_executor

def main():
    st.title("🤖 Reddit Reply Agent")
    st.markdown("A tool to find relevant posts about your product and generate natural-sounding replies")
    
    with st.sidebar:
        st.header("Configuration")
        
        st.subheader("API Keys")
        api_key = st.text_input("Groq API Key", value=st.session_state.GROQ_API_KEY, type="password")
        if api_key:
            st.session_state.GROQ_API_KEY = api_key
        
        st.subheader("Reddit Credentials")
        reddit_client_id = st.text_input("Client ID", value="", type="password")
        reddit_client_secret = st.text_input("Client Secret", value="", type="password")
        reddit_user_agent = st.text_input("User Agent", value="")
        reddit_username = st.text_input("Username", value="")
        reddit_password = st.text_input("Password", value="", type="password")

        st.subheader("About")
        st.markdown("This app uses LLMs to find relevant Reddit posts and generate authentic-sounding comments that are actually posted to Reddit.")
        
        # Warning about posting
        st.warning("⚠️ This app will actually post comments to Reddit using your credentials!")

    tab1, tab2 = st.tabs(["🔍 Find & Reply", "⚙️ Test Agent"])

    # Tab 1: Manual post finding and reply
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Your Product Information")
            product_name = st.text_input("Product Name", "StutterEase")
            product_description = st.text_area("Product Description", "An app that helps people practice speaking to overcome stuttering")
            target_keyword = st.text_input("Main Keyword", "stuttering")
            
            if st.button("Find Relevant Posts", key="find_posts"):
                if not api_key:
                    st.error("Please enter your Groq API Key in the sidebar")
                else:
                    with st.spinner("Finding relevant subreddits and posts..."):
                        # Initialize Reddit
                        try:
                            reddit = initialize_reddit(
                                client_id=reddit_client_id,
                                client_secret=reddit_client_secret,
                                user_agent=reddit_user_agent,
                                username=reddit_username,
                                password=reddit_password
                            )
                            
                            # Get subreddits
                            user_input = f"{product_name} is {product_description}. It helps with {target_keyword}."
                            subreddits = extract_subreddits_with_groq(user_input, api_key)
                            
                            # Get posts
                            posts = fetch_relevant_posts(reddit, subreddits, target_keyword)
                            
                            # Filter posts
                            filtered_posts = filter_posts_with_llm(user_input, target_keyword, posts, api_key)
                            
                            # Store in session state
                            st.session_state.found_posts = filtered_posts
                            st.session_state.product_info = user_input
                            
                            st.success(f"Found {len(filtered_posts)} relevant posts")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
        
        with col2:
            st.subheader("Relevant Reddit Posts")
            
            if "found_posts" in st.session_state and st.session_state.found_posts:
                for i, post in enumerate(st.session_state.found_posts):
                    with st.expander(f"📝 {post['title']} (r/{post['subreddit']})"):
                        st.write(f"**Subreddit:** r/{post['subreddit']}")
                        st.write(f"**Score:** {post['score']}")
                        st.write(f"**Comments:** {post['num_comments']}")
                        st.write(f"**Link:** [View on Reddit]({post['url']})")
                        st.write(f"**Post ID:** {post['post_id']}")
                        
                        if st.button("Generate Reply", key=f"reply_{i}"):
                            with st.spinner("Generating reply..."):
                                try:
                                    reddit = initialize_reddit(
                                        client_id=reddit_client_id,
                                        client_secret=reddit_client_secret,
                                        user_agent=reddit_user_agent,
                                        username=reddit_username,
                                        password=reddit_password
                                    )
                                    
                                    post_obj = reddit.submission(id=post['post_id'])
                                    
                                    comment_text = generate_comment(
                                        user_input=st.session_state.product_info,
                                        post_title=post['title'],
                                        post_content=post_obj.selftext if hasattr(post_obj, 'selftext') and post_obj.selftext else "No additional text available.",
                                        api_key=api_key
                                    )
                                    
                                    st.text_area("Generated Reply", comment_text, height=150)
                                    
                                    post_option = st.selectbox("Would you like to post this comment?", ["No, just show me the draft", "Yes, post this comment"], key=f"post_option_{i}")
                                    
                                    if post_option == "Yes, post this comment" and st.button("Confirm Post", key=f"confirm_post_{i}"):
                                        # Actually post to Reddit
                                        post_obj.reply(comment_text)
                                        st.success("Comment posted successfully to Reddit!")
                                except Exception as e:
                                    st.error(f"Error: {str(e)}")
            else:
                st.info("No posts found yet. Use the form on the left to search for relevant posts.")

    # Tab 2: Test Agent directly with real-time feedback
    with tab2:
        st.subheader("Test the Reddit Reply Agent")
        
        user_query = st.text_area(
            "Enter your instruction for the agent", 
            "My product StutterEase helps people with stuttering. I want to find and comment on posts about stuttering.",
            height=100
        )
        
        if st.button("Run Agent"):
            if not api_key:
                st.error("Please enter your Groq API Key in the sidebar")
            else:
                thought_container = st.empty()
                
                try:
                    # Initialize Reddit
                    reddit = initialize_reddit(
                        client_id=reddit_client_id,
                        client_secret=reddit_client_secret,
                        user_agent=reddit_user_agent,
                        username=reddit_username,
                        password=reddit_password
                    )
                    
                    # Create a callback handler for streaming thoughts
                    callback_handler = StreamingCallbackHandler(thought_container)
                    
                    # Setup agent with callback handler
                    agent = setup_agent(api_key, reddit, callback_handler)
                    
                    # Run the agent and display progress in real-time
                    with st.spinner("Agent is working... Watch real-time progress below"):
                        result = agent.invoke({"input": user_query})
                        
                    # Display final result
                    st.success("Agent completed successfully!")
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()