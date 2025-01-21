from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

# Check our tools documentation for more information on how to use them
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from pydantic import BaseModel, Field
from typing import List, Optional

class Activity(BaseModel):
    name: str = Field(..., description="Name of the activity")
    location: str = Field(..., description="Location of the activity")
    description: str = Field(..., description="Description of the activity")
    date: str = Field(..., description="Date of the activity")
    cousine: str = Field(..., description="Cousine of the restaurant")
    why_its_suitable: str = Field(..., description="Why it's suitable for the traveler")
    reviews: Optional[List[str]] = Field(..., description="List of reviews")
    rating: Optional[float] = Field(..., description="Rating of the activity")

@CrewBase
class NewsletterCrew():
    """This crew will generate a newsletter"""
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['research_agent'],
            tools=[SerperDevTool(), ScrapeWebsiteTool()], # Example of custom tool, loaded at the beginning of file
            verbose=True,
            allow_delegation=False,
        )

    @agent
    def editor(self) -> Agent:
        return Agent(
            config=self.agents_config['editor_agent'],
            tools=[SerperDevTool(), ScrapeWebsiteTool()], # Example of custom tool, loaded at the beginning of file
            verbose=True,
            allow_delegation=False,
        )

    @agent
    def seo_expert(self) -> Agent:
        return Agent(
            config=self.agents_config['seo_agent'],
            verbose=True,
            allow_delegation=False,
        )

    @agent
    def copy_editor(self) -> Agent:
        return Agent(
            config=self.agents_config['personalized_activity_planner'],
            verbose=True,
            allow_delegation=False,
        )

    @agent
    def image_generator(self) -> Agent:
        return Agent(
            config=self.agents_config['image_generator'],
            verbose=True,
            allow_delegation=False,
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'],
            agent=self.researcher()
        )

    @task
    def editing_task(self) -> Task:
        return Task(
            config=self.tasks_config['newsletter_compilation_task'],
            agent=self.editor()
        )

    @task
    def seo_optimization_task(self) -> Task:
        return Task(
            config=self.tasks_config['seo_optimization_task'],
            agent=self.seo_expert()
        )

    @task
    def copy_editing_task(self) -> Task:
        return Task(
            config=self.tasks_config['copy_editing_task'],
            agent=self.copy_editor()
        )

    @task
    def image_generation_task(self) -> Task:
        return Task(
            config=self.tasks_config['image_generation_task'],
            agent=self.image_generator()
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Newsletter crew"""
        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=2,
        )
