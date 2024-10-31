from __future__ import annotations

from typing import Any, override

import click
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI
from pangea import PangeaConfig
from pangea.services import AuthZ
from pangea.services.authz import Resource, Subject
from pydantic import SecretStr

load_dotenv(override=True)


PROMPT = PromptTemplate.from_template(
    """Answer the following questions as best you can. You have access to the following tools:

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
)


class SecretStrParamType(click.ParamType):
    name = "secret"

    @override
    def convert(self, value: Any, param: click.Parameter | None = None, ctx: click.Context | None = None) -> SecretStr:
        if isinstance(value, SecretStr):
            return value

        return SecretStr(value)


SECRET_STR = SecretStrParamType()


@click.command()
@click.option("--user", required=True, help="Unique username to simulate retrieval as.")
@click.option(
    "--authz-token",
    envvar="PANGEA_AUTHZ_TOKEN",
    type=SECRET_STR,
    required=True,
    help="Pangea AuthZ API token. May also be set via the `PANGEA_AUTHZ_TOKEN` environment variable.",
)
@click.option(
    "--pangea-domain",
    envvar="PANGEA_DOMAIN",
    default="aws.us.pangea.cloud",
    show_default=True,
    required=True,
    help="Pangea API domain. May also be set via the `PANGEA_DOMAIN` environment variable.",
)
@click.option("--model", default="gpt-4o-mini", show_default=True, required=True, help="OpenAI model.")
@click.option(
    "--openai-api-key",
    envvar="OPENAI_API_KEY",
    type=SECRET_STR,
    required=True,
    help="OpenAI API key. May also be set via the `OPENAI_API_KEY` environment variable.",
)
@click.argument("prompt")
def main(
    *,
    prompt: str,
    user: str,
    authz_token: SecretStr,
    pangea_domain: str,
    model: str,
    openai_api_key: SecretStr,
) -> None:
    authz = AuthZ(token=authz_token.get_secret_value(), config=PangeaConfig(domain=pangea_domain))
    subject = Subject(type="user", id=user)

    # Check if user is authorized to run the tool.
    response = authz.check(resource=Resource(type="duckduckgo"), action="read", subject=subject)
    if response.result is None or not response.result.allowed:
        click.echo(f"User {subject.id} is not authorized to use this tool.")
        return

    # Set up Pangea AuthZ + DuckDuckGo tool.
    ddg = DuckDuckGoSearchRun()
    tools = [ddg]
    llm = ChatOpenAI(model=model, api_key=openai_api_key, temperature=0)
    agent = create_react_agent(tools=tools, llm=llm, prompt=PROMPT)
    agent_executor = AgentExecutor(agent=agent, tools=tools)

    click.echo(agent_executor.invoke({"input": prompt})["output"])


if __name__ == "__main__":
    main()
