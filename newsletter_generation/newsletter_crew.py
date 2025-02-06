from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

# Check our tools documentation for more information on how to use them
from crewai_tools import SerperDevTool, ScrapeWebsiteTool

@CrewBase
class NewsletterCrew():
    """This crew will generate a newsletter"""
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['research_agent'],
            tools=[SerperDevTool(), ScrapeWebsiteTool()],
            verbose=True,
            allow_delegation=False,
        )

    @agent
    def editor(self) -> Agent:
        return Agent(
            config=self.agents_config['editor_agent'],
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
            config=self.agents_config['copy_editor_agent'],
            verbose=True,
            allow_delegation=False,
        )

    @agent
    def image_generator(self) -> Agent:
        return Agent(
            config=self.agents_config['image_generator_agent'],
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
            agents=self.agents,
            tasks=self.tasks,
            process=Process.hierarchical,
            manager_llm="gpt-4o",
            verbose=True
        )
