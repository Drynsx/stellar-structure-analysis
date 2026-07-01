"""SQLAlchemy models for PostgreSQL-backed stellar analysis results."""

from __future__ import annotations

from sqlalchemy import Column, Float, ForeignKey, Integer, String, create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.types import JSON


Base = declarative_base()
JsonProfile = JSON().with_variant(JSONB, "postgresql")


class Star(Base):
    __tablename__ = "stars"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, default="Custom Star")
    mass = Column(Float, nullable=False)
    teff = Column(Float, nullable=False)
    metallicity = Column(Float, nullable=False)
    age = Column(Float, nullable=False)
    source = Column(String(64), nullable=False, default="Observed")

    profiles = relationship("Profile", back_populates="star", cascade="all, delete-orphan")
    results = relationship("Result", back_populates="star", cascade="all, delete-orphan")


class Profile(Base):
    __tablename__ = "profiles"

    id = Column(Integer, primary_key=True, index=True)
    star_id = Column(Integer, ForeignKey("stars.id"), nullable=False, index=True)
    r_profile = Column(JsonProfile, nullable=False)
    rho_profile = Column(JsonProfile, nullable=False)
    p_profile = Column(JsonProfile, nullable=False)
    t_profile = Column(JsonProfile, nullable=False)

    star = relationship("Star", back_populates="profiles")


class Result(Base):
    __tablename__ = "results"

    id = Column(Integer, primary_key=True, index=True)
    star_id = Column(Integer, ForeignKey("stars.id"), nullable=False, index=True)
    n_global = Column(Float, nullable=False)
    n_core = Column(Float, nullable=True)
    n_rad = Column(Float, nullable=True)
    n_conv = Column(Float, nullable=True)
    delta_n_rad = Column(Float, nullable=False)
    delta_n_mu = Column(Float, nullable=False)
    delta_n_conv = Column(Float, nullable=False)
    delta_n_nuc = Column(Float, nullable=False)
    delta_n_deg = Column(Float, nullable=False)
    anomaly_score = Column(Float, nullable=False)
    status = Column(String(32), nullable=False, default="Normal")

    star = relationship("Star", back_populates="results")


def make_engine(database_url: str):
    return create_engine(database_url, pool_pre_ping=True, future=True)


def make_session_factory(database_url: str):
    engine = make_engine(database_url)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def create_tables(database_url: str) -> None:
    engine = make_engine(database_url)
    Base.metadata.create_all(bind=engine)
